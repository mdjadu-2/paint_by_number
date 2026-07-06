import os
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from skimage import graph as sk_graph
from scipy.ndimage import median_filter
from skimage.measure import label as sk_label
from sklearn.preprocessing import StandardScaler

def initialize_processing_canvas(image_path, target_w, target_h, scale_factor):
    """
    Loads the source image, handles color conversion, and creates
    the downscaled processing canvas with perfect aspect-ratio padding.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Missing image at: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Calculate dimensions for the smaller processing canvas
    proc_w = target_w // scale_factor
    proc_h = target_h // scale_factor
    
    # Determine the uniform scale to fit inside target bounds
    oh, ow = img_rgb.shape[:2]
    scale = min(proc_w / ow, proc_h / oh)
    new_w, new_h = int(ow * scale), int(oh * scale)
    
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Compute explicit white padding dimensions
    pad_w = proc_w - new_w
    pad_h = proc_h - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    
    img_proc = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    return img_proc

def preprocess_and_quantize(img_proc, k, d, sigma_color, sigma_space):
    """
    Smooths input noise while protecting structural edges, standardizes 
    LAB channels, and executes k-means to find the dominant image palette.
    """
    # Smooth out sensor noise and minor gradients
    smoothed = cv2.bilateralFilter(img_proc, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    
    # Convert to LAB color space
    img_lab = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB)
    h, w = img_lab.shape[:2]
    pixels_lab = img_lab.reshape(-1, 3).astype(np.float32)
    
    # Balance channel variances equally
    scaler = StandardScaler()
    pixels_scaled = scaler.fit_transform(pixels_lab)
    
    # Run clustering on the downscaled pixel array
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(pixels_scaled)
    
    # Reverse scaling transformation to obtain actual LAB colors
    centers_lab = scaler.inverse_transform(km.cluster_centers_)
    labels_2d = labels.reshape(h, w).astype(np.int32)
    
    return labels_2d, centers_lab

def form_initial_regions(labels_2d, k):
    """
    Segments contiguous color pixels into unique region IDs and builds
    the structural region-to-cluster lookup reference.
    """
    h, w = labels_2d.shape
    region_map = np.zeros_like(labels_2d, dtype=np.int32)
    next_id = 1
    
    for cluster_id in range(k):
        mask = (labels_2d == cluster_id)
        blobs = sk_label(mask, connectivity=2)
        n = blobs.max()
        region_map[mask] = blobs[mask] + next_id
        next_id += n

    # Build the 1D region-to-cluster lookup vector
    region_to_cluster = np.zeros(region_map.max() + 1, dtype=np.int32)
    region_to_cluster[region_map.ravel()] = labels_2d.ravel()
    
    print(f"Step 3 -> Total unique regions formed: {len(np.unique(region_map))}")
    return region_map, region_to_cluster

def merge_regions_advanced(region_map, region_to_cluster, centers_lab, dpi, scale_factor, min_paint_mm, contrast_threshold, hard_floor):
    """
    Calculates physical constraints scaled to the low-res canvas and executes
    the advanced merge loop using 1D lookup array vectorization.
    """
    region_map = region_map.copy()
    max_id = region_map.max()
    lookup = np.arange(max_id + 1, dtype=np.int32)
    
    # Calculate physical boundaries scaled to the processing canvas
    px_per_mm = dpi / 25.4
    min_area_target = int((min_paint_mm * px_per_mm) ** 2)
    min_thickness_target = min_paint_mm * px_per_mm

    min_area_proc = max(5, min_area_target // (scale_factor ** 2))
    min_thickness_proc = max(1.5, min_thickness_target / scale_factor)
    
    print(f"Step 4 -> Processing Canvas Bounds -> Min Area: {min_area_proc}px, Min Thickness: {min_thickness_proc:.2f}px")

    while True:
        ids, counts = np.unique(region_map, return_counts=True)
        area = dict(zip(ids, counts))
        
        candidates = [r for r in ids if area[r] < min_area_proc * 3]
        if not candidates:
            break
            
        rag = sk_graph.RAG(region_map, connectivity=2)
        changed = False

        for r in candidates:
            if r not in rag: 
                continue
                
            neighbors = list(rag.neighbors(r))
            if not neighbors:
                continue

            r_cluster = region_to_cluster[r]
            r_col = centers_lab[r_cluster]
            
            def col_dist(n):
                return np.linalg.norm(r_col - centers_lab[region_to_cluster[n]])
            
            target = min(neighbors, key=col_dist)
            local_contrast = col_dist(target)

            is_too_small = area[r] < min_area_proc
            is_microscopic = area[r] <= hard_floor
            has_high_contrast = local_contrast > contrast_threshold

            is_too_thin = False
            if not is_too_small and not is_microscopic and not has_high_contrast:
                ys, xs = np.where(region_map == r)
                if len(ys) == 0: continue
                y_min, y_max = ys.min(), ys.max()
                x_min, x_max = xs.min(), xs.max()
                
                mask_crop = np.zeros((y_max - y_min + 3, x_max - x_min + 3), dtype=np.uint8)
                mask_crop[ys - y_min + 1, xs - x_min + 1] = 255
                
                dist_map = cv2.distanceTransform(mask_crop, cv2.DIST_L2, 3)
                is_too_thin = (np.max(dist_map) * 2) < min_thickness_proc

            should_merge = False
            if is_microscopic:
                should_merge = True
            elif has_high_contrast:
                should_merge = False
            elif is_too_small or is_too_thin:
                should_merge = True
            # Corrected Logic: Physical limits now have absolute veto power over color contrast
            # if is_microscopic or is_too_small or is_too_thin:
            #     should_merge = True  # If a human can't paint it, it dies. Period.
            # elif has_high_contrast:
            #     should_merge = False # Only protect details that are physically large enough to paint
            # else:
            #     should_merge = True  # Merge low-contrast baseline areas normally

            if should_merge:
                lookup[lookup == r] = target
                region_to_cluster[r] = region_to_cluster[target]
                area[target] = area.get(target, 0) + area[r]
                rag.remove_node(r)
                changed = True

        if not changed:
            break
            
        region_map = lookup[region_map]
            
    print(f"Step 4 -> Regions remaining after advanced merge: {len(np.unique(region_map))}")
    return region_map

def smooth_label_boundaries(region_map):
    """
    Cleans up isolated pixel necks and jagged boundary noise on the label map
    by replacing isolated pixels with their majority neighbor.
    """
    smoothed = region_map.copy()
    
    # 4-connectivity neighbor shifts to detect isolated single-pixel structures
    up = np.roll(region_map, -1, axis=0)
    down = np.roll(region_map, 1, axis=0)
    left = np.roll(region_map, -1, axis=1)
    right = np.roll(region_map, 1, axis=1)
    
    # Identify pixels that do not match any of their immediate neighbors
    isolated_mask = (region_map != up) & (region_map != down) & (region_map != left) & (region_map != right)
    
    # Snap isolated noise pixels directly to their top neighbor's region ID
    smoothed[isolated_mask] = up[isolated_mask]
    
    print(f"Step 5 -> Regions remaining after boundary smoothing: {len(np.unique(smoothed))}")
    return smoothed

def generate_final_outputs(merged_proc, region_to_cluster, centers_lab, target_w, target_h, scale_factor, font_scale, text_color, output_dir):
    """
    Upscales the label map, runs high-res anti-aliasing to smooth out 90-degree steps,
    rebuilds the colored reference image, and creates the numbered outline canvas.
    """
    print("Upscaling label map to high resolution...")
    merged_target = cv2.resize(merged_proc, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    print("Smoothing high-resolution boundaries (Anti-Aliasing)...")
    # merged_target_float = merged_target.astype(np.float32)
    # smoothed_target_float = cv2.medianBlur(merged_target_float, ksize=11)
    # merged_target = smoothed_target_float.astype(np.int32)
    merged_target = median_filter(merged_target, size=11)

    # --- Rebuild Colored Reference Image ---
    print("Rebuilding colored reference image...")
    cluster_per_pixel = region_to_cluster[merged_target]
    merged_lab = centers_lab[cluster_per_pixel].reshape(target_h, target_w, 3).astype(np.uint8)
    merged_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)

    ref_path = os.path.join(output_dir, "merged_output_hd_v4.png")
    cv2.imwrite(ref_path, cv2.cvtColor(merged_rgb, cv2.COLOR_RGB2BGR))
    print(f"Saved clean reference image to: {ref_path}")
    print(f"Final unique region count on HD canvas: {len(np.unique(merged_target))}")

    # --- Generate Blank Outline Canvas Map ---
    print("Generating outline canvas...")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(merged_target.astype(np.float32), kernel)
    eroded = cv2.erode(merged_target.astype(np.float32), kernel)
    edge_mask = dilated != eroded

    canvas_map = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    canvas_map[edge_mask] = [0, 0, 0]

    print("Calculating optimal region centers and placing numbers...")
    unique_regions = np.unique(merged_proc)

    for r in unique_regions:
        ys, xs = np.where(merged_proc == r)
        if len(ys) == 0: 
            continue
            
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        
        mask_crop = np.zeros((y_max - y_min + 3, x_max - x_min + 3), dtype=np.uint8)
        mask_crop[ys - y_min + 1, xs - x_min + 1] = 255
        
        dist_map = cv2.distanceTransform(mask_crop, cv2.DIST_L2, 3)
        _, _, _, max_loc = cv2.minMaxLoc(dist_map)
        
        cx_small = x_min - 1 + max_loc[0]
        cy_small = y_min - 1 + max_loc[1]
        
        cx_target = int(cx_small * scale_factor + scale_factor // 2)
        cy_target = int(cy_small * scale_factor + scale_factor // 2)
        
        cluster_id = region_to_cluster[r]
        text = str(cluster_id)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 1
        
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = cx_target - text_w // 2
        text_y = cy_target + text_h // 2
        
        if 0 <= text_x < target_w and 0 <= text_y < target_h:
            cv2.putText(canvas_map, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    canvas_path = os.path.join(output_dir, "paint_by_numbers_canvas.png")
    pdf_path = os.path.join(output_dir, "paint_by_numbers_canvas.pdf")  
    cv2.imwrite(canvas_path, canvas_map)
    print(f"Saved printable line art to: {canvas_path}")
    
    print(f"Converting {canvas_path} to PDF...")
    
    # Open the image and ensure it is in RGB mode for PDF compliance
    img = Image.open(canvas_path)
    img_rgb = img.convert('RGB')
    
    # Save directly as a PDF
    img_rgb.save(pdf_path, "PDF")
    print(f"Success! Saved printable PDF to: {pdf_path}")






























