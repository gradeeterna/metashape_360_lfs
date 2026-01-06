#!/usr/bin/env python3
"""
Standalone Metashape to LichtFeld Converter

Converts Metashape XML camera poses + PLY point cloud to LichtFeld-compatible
transforms.json format, without requiring nerfstudio.

Dependencies:
    pip install numpy open3d

Usage:
    python metashape_to_lichtfeld.py --images ./images/ --xml cameras.xml --ply sparse.ply --output ./output/

Based on nerfstudio's metashape_utils.py (Apache 2.0 License)
"""

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

# Try to import open3d, fall back to plyfile if not available
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    try:
        from plyfile import PlyData, PlyElement
        HAS_PLYFILE = True
    except ImportError:
        HAS_PLYFILE = False


def find_param(calib_xml: ET.Element, param_name: str) -> float:
    """Find a parameter in calibration XML, return 0.0 if not found."""
    param = calib_xml.find(param_name)
    if param is not None and param.text:
        return float(param.text)
    return 0.0


def parse_metashape_xml(xml_path: Path) -> Dict[str, Any]:
    """
    Parse Metashape XML file to extract sensors and camera transforms.
    
    Returns:
        Dictionary containing sensor_dict, camera_model, frames list
    """
    xml_tree = ET.parse(xml_path)
    root = xml_tree.getroot()
    chunk = root[0]
    sensors = chunk.find("sensors")
    
    if sensors is None:
        raise ValueError("No sensors found in Metashape XML")
    
    # Find calibrated sensors
    calibrated_sensors = [
        sensor for sensor in sensors.iter("sensor")
        if sensor.get("type") == "spherical" or sensor.find("calibration")
    ]
    
    if not calibrated_sensors:
        raise ValueError("No calibrated sensor found in Metashape XML")
    
    # Check sensor types are consistent
    sensor_types = [s.get("type") for s in calibrated_sensors]
    if sensor_types.count(sensor_types[0]) != len(sensor_types):
        raise ValueError(
            "All Metashape sensors must have the same type. "
            "Mixed camera types are not supported."
        )
    
    # Map Metashape sensor type to camera model string
    sensor_type = sensor_types[0]
    if sensor_type == "frame":
        camera_model = "PINHOLE"
    elif sensor_type == "fisheye":
        camera_model = "OPENCV_FISHEYE"
    elif sensor_type == "spherical":
        camera_model = "EQUIRECTANGULAR"
    else:
        raise ValueError(f"Unsupported Metashape sensor type: {sensor_type}")
    
    # Parse sensor calibration data
    sensor_dict = {}
    for sensor in calibrated_sensors:
        s = {}
        resolution = sensor.find("resolution")
        if resolution is None:
            raise ValueError("Resolution not found in Metashape XML")
        
        s["w"] = int(resolution.get("width"))
        s["h"] = int(resolution.get("height"))
        
        calib = sensor.find("calibration")
        if calib is None:
            # Spherical sensors may not have calibration
            if sensor_type == "spherical":
                s["fl_x"] = s["w"] / 2.0
                s["fl_y"] = s["h"]
                s["cx"] = s["w"] / 2.0
                s["cy"] = s["h"] / 2.0
            else:
                raise ValueError(f"No calibration found for sensor {sensor.get('id')}")
        else:
            f = calib.find("f")
            if f is None or f.text is None:
                raise ValueError("Focal length not found in Metashape XML")
            s["fl_x"] = s["fl_y"] = float(f.text)
            s["cx"] = find_param(calib, "cx") + s["w"] / 2.0
            s["cy"] = find_param(calib, "cy") + s["h"] / 2.0
            
            # Distortion parameters
            s["k1"] = find_param(calib, "k1")
            s["k2"] = find_param(calib, "k2")
            s["k3"] = find_param(calib, "k3")
            s["k4"] = find_param(calib, "k4")
            s["p1"] = find_param(calib, "p1")
            s["p2"] = find_param(calib, "p2")
        
        sensor_dict[sensor.get("id")] = s
    
    # Parse component transforms (for multi-chunk projects)
    components = chunk.find("components")
    component_dict = {}
    if components is not None:
        for component in components.iter("component"):
            transform = component.find("transform")
            if transform is not None:
                rotation = transform.find("rotation")
                if rotation is None or rotation.text is None:
                    r = np.eye(3)
                else:
                    r = np.array([float(x) for x in rotation.text.split()]).reshape((3, 3))
                
                translation = transform.find("translation")
                if translation is None or translation.text is None:
                    t = np.zeros(3)
                else:
                    t = np.array([float(x) for x in translation.text.split()])
                
                scale = transform.find("scale")
                if scale is None or scale.text is None:
                    s = 1.0
                else:
                    s = float(scale.text)
                
                m = np.eye(4)
                m[:3, :3] = r
                m[:3, 3] = t / s
                component_dict[component.get("id")] = m
    
    # Parse camera frames
    cameras = chunk.find("cameras")
    if cameras is None:
        raise ValueError("No cameras found in Metashape XML")
    
    return {
        "sensor_dict": sensor_dict,
        "component_dict": component_dict,
        "cameras": cameras,
        "camera_model": camera_model
    }


def transform_camera_matrix(transform: np.ndarray, fix_upside_down: bool = True) -> np.ndarray:
    """
    Convert Metashape camera transform to LichtFeld/nerfstudio convention.
    
    Args:
        transform: 4x4 camera-to-world matrix from Metashape
        fix_upside_down: If True, apply additional 180° rotation to fix upside-down scene
    
    Returns:
        Transformed 4x4 matrix
    """
    # Metashape: camera looks at -Z, +X right, +Y up
    # Step 1: Rotate scene according to nerfstudio convention (row swap)
    transform = transform[[2, 0, 1, 3], :]
    
    # Step 2: Convert from OpenCV to OpenGL (flip Y and Z columns)
    transform[:, 1:3] *= -1
    
    # Step 3: Fix orientation with +90° rotation around X-axis
    # This converts from bottom-up view to natural ground-level view
    if fix_upside_down:
        # Rotation matrix around X by +90°: [[1,0,0], [0,0,-1], [0,1,0]]
        cos_90 = 0.0   # cos(90°) = 0
        sin_90 = 1.0   # sin(90°) = 1
        rot_x_pos90 = np.array([
            [1, 0, 0, 0],
            [0, cos_90, -sin_90, 0],  # [0, 0, -1, 0]
            [0, sin_90, cos_90, 0],   # [0, 1, 0, 0]
            [0, 0, 0, 1]
        ], dtype=np.float64)
        transform = rot_x_pos90 @ transform
    
    # Step 4: Pre-compensate for LichtFeld's 180° Y-rotation
    # LichtFeld applies a Y-rotation to convert from OpenGL to COLMAP convention
    # We apply the inverse here so they cancel out
    cos_pi = -1.0  # cos(180°) = -1
    sin_pi = 0.0   # sin(180°) = 0
    y_rot_180 = np.array([
        [cos_pi, 0, sin_pi, 0],
        [0, 1, 0, 0],
        [-sin_pi, 0, cos_pi, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    transform = y_rot_180 @ transform
    
    return transform



def get_applied_transform(fix_upside_down: bool = True) -> np.ndarray:
    """
    Get the 3x4 transformation matrix applied to point cloud.
    
    Note: LichtFeld only applies Y-rotation to cameras, not point clouds.
    So we don't need Y-rotation pre-compensation here.
    """
    # Base transform: row swap [2, 0, 1]
    applied_transform = np.eye(4)[:3, :]
    applied_transform = applied_transform[np.array([2, 0, 1]), :]
    
    # Add orientation fix: +90° rotation around X
    if fix_upside_down:
        cos_90 = 0.0
        sin_90 = 1.0
        rot_x_pos90 = np.array([
            [1, 0, 0],
            [0, cos_90, -sin_90],  # [0, 0, -1]
            [0, sin_90, cos_90]    # [0, 1, 0]
        ], dtype=np.float64)
        applied_transform[:3, :3] = rot_x_pos90 @ applied_transform[:3, :3]
    
    # NOTE: No Y-rotation here - LichtFeld only applies Y-rot to cameras, not point clouds
    
    return applied_transform




def convert_metashape_to_lichtfeld(
    images_dir: Path,
    xml_path: Path,
    output_dir: Optional[Path] = None,
    ply_path: Optional[Path] = None,
    fix_upside_down: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convert Metashape data to LichtFeld-compatible transforms.json format.
    
    Args:
        images_dir: Directory containing images
        xml_path: Path to Metashape cameras.xml
        output_dir: Output directory (defaults to same directory as xml_path)
        ply_path: Optional path to point cloud PLY file
        fix_upside_down: If True, fix the upside-down scene orientation
        verbose: Print progress messages
    
    Returns:
        Dictionary with conversion statistics
    """
    # Default output to same directory as XML
    if output_dir is None:
        output_dir = xml_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Parsing Metashape XML: {xml_path}")
    
    # Parse XML
    xml_data = parse_metashape_xml(xml_path)
    sensor_dict = xml_data["sensor_dict"]
    component_dict = xml_data["component_dict"]
    cameras_xml = xml_data["cameras"]
    camera_model = xml_data["camera_model"]
    
    if verbose:
        print(f"Camera model: {camera_model}")
        print(f"Found {len(sensor_dict)} sensor(s)")
    
    # Build image filename map
    image_extensions = [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))
        image_files.extend(images_dir.glob(f"*{ext.upper()}"))
    
    image_filename_map = {}
    for img_path in image_files:
        # Map both with and without extension
        image_filename_map[img_path.stem] = img_path
        image_filename_map[img_path.name] = img_path
    
    if verbose:
        print(f"Found {len(image_files)} images in {images_dir}")
    
    # Process frames
    frames = []
    num_skipped = 0
    
    for camera in cameras_xml.iter("camera"):
        camera_label = camera.get("label")
        if not camera_label:
            continue
        
        # Find matching image
        if camera_label not in image_filename_map:
            # Try without extension
            camera_label_no_ext = camera_label.split(".")[0]
            if camera_label_no_ext not in image_filename_map:
                if verbose:
                    print(f"  Skipping {camera.get('label')}: no matching image")
                num_skipped += 1
                continue
            camera_label = camera_label_no_ext
        
        # Get sensor data
        sensor_id = camera.get("sensor_id")
        if sensor_id not in sensor_dict:
            if verbose:
                print(f"  Skipping {camera.get('label')}: no sensor calibration")
            num_skipped += 1
            continue
        
        # Get camera transform
        transform_elem = camera.find("transform")
        if transform_elem is None or transform_elem.text is None:
            if verbose:
                print(f"  Skipping {camera.get('label')}: no transform")
            num_skipped += 1
            continue
        
        transform = np.array([float(x) for x in transform_elem.text.split()]).reshape((4, 4))
        
        # Apply component transform if present
        component_id = camera.get("component_id")
        if component_id in component_dict:
            transform = component_dict[component_id] @ transform
        
        # Convert to LichtFeld convention
        transform = transform_camera_matrix(transform, fix_upside_down)
        
        # Get image path relative to output directory
        src_image = image_filename_map[camera_label]
        try:
            rel_path = src_image.resolve().relative_to(output_dir.resolve())
            file_path = rel_path.as_posix()
        except ValueError:
            # Images not in output_dir subtree - use images/filename format if images_dir is named "images"
            if images_dir.name == "images":
                file_path = f"images/{src_image.name}"
            else:
                # Use absolute path as fallback
                file_path = src_image.resolve().as_posix()
        
        # Build frame data
        frame = {
            "file_path": file_path,
            "transform_matrix": transform.tolist()
        }
        frame.update(sensor_dict[sensor_id])
        frames.append(frame)
    
    if verbose:
        print(f"Processed {len(frames)} camera frames")
        if num_skipped > 0:
            print(f"Skipped {num_skipped} cameras")
    
    # Build output data
    data = {
        "camera_model": camera_model,
        "frames": frames
    }
    
    # Store applied transform for reference
    applied_transform = get_applied_transform(fix_upside_down)
    data["applied_transform"] = applied_transform.tolist()
    
    # Process point cloud
    if ply_path is not None and ply_path.exists():
        if verbose:
            print(f"Processing point cloud: {ply_path}")
        
        if HAS_OPEN3D:
            pc = o3d.io.read_point_cloud(str(ply_path))
            points3D = np.asarray(pc.points)
            
            # Apply same transformation as cameras
            points3D = np.einsum("ij,bj->bi", applied_transform[:3, :3], points3D) + applied_transform[:3, 3]
            pc.points = o3d.utility.Vector3dVector(points3D)
            
            output_ply = output_dir / "pointcloud.ply"
            o3d.io.write_point_cloud(str(output_ply), pc)
            data["ply_file_path"] = "pointcloud.ply"
            pointcloud_written = True
            
            if verbose:
                print(f"Wrote point cloud with {len(points3D)} points to {output_ply}")
        
        elif HAS_PLYFILE:
            # Fallback to plyfile
            plydata = PlyData.read(str(ply_path))
            vertex = plydata['vertex']
            points3D = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
            
            # Apply transformation
            points3D = np.einsum("ij,bj->bi", applied_transform[:3, :3], points3D) + applied_transform[:3, 3]
            
            # Write back (simplified, may lose color data)
            new_vertex = np.zeros(len(points3D), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            new_vertex['x'] = points3D[:, 0]
            new_vertex['y'] = points3D[:, 1]
            new_vertex['z'] = points3D[:, 2]
            
            output_ply = output_dir / "pointcloud.ply"
            PlyData([PlyElement.describe(new_vertex, 'vertex')]).write(str(output_ply))
            data["ply_file_path"] = "pointcloud.ply"
            pointcloud_written = True
            
            if verbose:
                print(f"Wrote point cloud with {len(points3D)} points (colors may be lost)")
        else:
            print("WARNING: Neither open3d nor plyfile installed. Skipping point cloud.")
            print("         Install with: pip install open3d")
            pointcloud_written = False
    else:
        pointcloud_written = False
    
    # Write transforms.json
    output_json = output_dir / "transforms.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    
    if verbose:
        print(f"\nWrote {output_json}")
        print(f"Dataset ready: {len(frames)} frames, camera_model={camera_model}")
    
    return {
        "num_frames": len(frames),
        "num_skipped": num_skipped,
        "camera_model": camera_model,
        "has_pointcloud": pointcloud_written
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert Metashape XML + PLY to LichtFeld transforms.json format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python metashape_to_lichtfeld.py --images ./images/ --xml cameras.xml
    python metashape_to_lichtfeld.py --images ./images/ --xml cameras.xml --ply sparse.ply
    python metashape_to_lichtfeld.py --images ./images/ --xml cameras.xml --ply sparse.ply --output ./other/
        """
    )
    
    parser.add_argument("--images", type=Path, required=True,
                        help="Directory containing source images")
    parser.add_argument("--xml", type=Path, required=True,
                        help="Path to Metashape cameras.xml file")
    parser.add_argument("--ply", type=Path, default=None,
                        help="Optional path to point cloud PLY file")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (defaults to same folder as XML)")
    parser.add_argument("--no-fix-rotation", action="store_true",
                        help="Disable 180° rotation fix (scene may appear upside-down)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    
    args = parser.parse_args()
    
    if not args.images.is_dir():
        print(f"Error: Images directory not found: {args.images}")
        return 1
    
    if not args.xml.is_file():
        print(f"Error: XML file not found: {args.xml}")
        return 1
    
    if args.ply and not args.ply.is_file():
        print(f"Error: PLY file not found: {args.ply}")
        return 1
    
    try:
        result = convert_metashape_to_lichtfeld(
            images_dir=args.images,
            xml_path=args.xml,
            output_dir=args.output,
            ply_path=args.ply,
            fix_upside_down=not args.no_fix_rotation,
            verbose=not args.quiet
        )
        
        if not args.quiet:
            print("\nConversion complete!")
            print(f"  Frames: {result['num_frames']}")
            print(f"  Skipped: {result['num_skipped']}")
            print(f"  Camera model: {result['camera_model']}")
            print(f"  Point cloud: {'Yes' if result['has_pointcloud'] else 'No'}")
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
