import os
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from skimage.measure import label as sk_label
from skimage import graph as sk_graph

def initialize_adobe_canvas(image_path, target_w, target_h, scale_factor):
    """
    Ingests source asset, processes alpha-channel transparency cleanly against white,
    and returns an unpadded raw processing canvas layout.
    """
    img_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise FileNotFoundError(f"Missing image at: {image_path}")
    
    if len(img_bgr.shape) == 4 or img_bgr.shape[2] == 4:
        alpha = img_bgr[:, :, 3] / 255.0
        bgr = img_bgr[:, :, :3]
        white_bg = np.ones_like(bgr, dtype=np.uint8) * 255
        img_bgr = (bgr * alpha[:, :, np.newaxis] + white_bg * (1.0 - alpha[:, :, np.newaxis])).astype(np.uint8)
    else:
        img_bgr = img_bgr[:, :, :3]

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    proc_w = target_w // scale_factor
    proc_h = target_h // scale_factor
    
    oh, ow = img_rgb.shape[:2]
    scale = min(proc_w / ow, proc_h / oh)
    new_w, new_h = int(ow * scale), int(oh * scale)
    
    img_proc = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return img_proc

def process_adobe_slic(img_proc, k_colors=16, compactness=5, n_segments=1500):
    """
    MIDDLE-GROUND ENGINE: Employs a dual-stage texture melting pass to destroy hair/shirt noise 
    while protecting anatomical structures via a tight, optimized size floor window.
    """
    # 1. Heavy texture melting pass to scrub out high-frequency beard strands and shirt lines
    blurred_rgb = cv2.GaussianBlur(img_proc, (5, 5), 0)
    smoothed_rgb = cv2.bilateralFilter(blurred_rgb, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Extract baseline structural lines to locate key facial features
    gray_proc = cv2.cvtColor(smoothed_rgb, cv2.COLOR_RGB2GRAY)
    feature_saliency = cv2.Canny(gray_proc, 30, 100)
    
    # 2. Pyramidal Mean Shift to smooth remaining skin variations into clean blocks
    img_bgr = cv2.cvtColor(smoothed_rgb, cv2.COLOR_RGB2BGR)
    shifted_bgr = cv2.pyrMeanShiftFiltering(img_bgr, sp=6, sr=8, maxLevel=2)
    shifted_rgb = cv2.cvtColor(shifted_bgr, cv2.COLOR_BGR2RGB)
    
    # 3. Cluster colors in LAB space to preserve natural brightness values
    h, w = shifted_rgb.shape[:2]
    img_lab = cv2.cvtColor(shifted_rgb, cv2.COLOR_RGB2LAB)
    pixels_lab = img_lab.reshape(-1, 3).astype(np.float32)
    
    km = KMeans(n_clusters=k_colors, n_init=10, random_state=42)
    color_labels = km.fit_predict(pixels_lab)
    centers_lab = km.cluster_centers_
    
    quantized_labels = color_labels.reshape(h, w).astype(np.int32)
    spatial_labels = sk_label(quantized_labels, connectivity=1, background=-1)
    
    n_regions = spatial_labels.max()
    region_to_cluster = np.zeros(n_regions + 1, dtype=np.int32)
    region_to_cluster[spatial_labels.ravel()] = quantized_labels.ravel()
    
    # 4. Map feature density to region keys
    ids = np.unique(spatial_labels)
    region_is_protected = {}
    for r in ids:
        if r == 0: continue
        mask = (spatial_labels == r)
        region_is_protected[r] = bool(np.any(feature_saliency[mask]))
        
    print(f"[SOTA Engine] Raw segments: {n_regions}. Executing texture-safe middle merge...")
    
    # 5. FIXED REGION WINDOW LOOP: Cleans noise while keeping eye shapes locked
    while True:
        ids, counts = np.unique(spatial_labels, return_counts=True)
        area = dict(zip(ids, counts))
        
        # Middle Ground: Details held to a paintable 35px floor; flat zones merge up to 90px
        candidates = [r for r in ids if r != 0 and area[r] < (35 if region_is_protected.get(r, False) else 90)]
        if not candidates:
            break
            
        rag = sk_graph.RAG(spatial_labels, connectivity=1)
        changed = False
        
        for r in candidates:
            if r not in rag: 
                continue
            neighbors = list(rag.neighbors(r))
            if not neighbors: 
                continue
                
            r_col = centers_lab[region_to_cluster[r]]
            def get_perceptual_dist(n):
                return np.linalg.norm(r_col - centers_lab[region_to_cluster[n]])
                
            target = min(neighbors, key=get_perceptual_dist)
            
            lookup = np.arange(spatial_labels.max() + 1, dtype=np.int32)
            lookup[lookup == r] = target
            spatial_labels = lookup[spatial_labels]
            rag.remove_node(r)
            changed = True
            
        if not changed: 
            break

    # Re-index remaining clean labels consecutively
    final_ids = np.unique(spatial_labels)
    remapped_labels = np.zeros_like(spatial_labels)
    final_region_to_cluster = np.zeros(len(final_ids) + 1, dtype=np.int32)
    
    for new_idx, old_id in enumerate(final_ids, start=1):
        remapped_labels[spatial_labels == old_id] = new_idx
        final_region_to_cluster[new_idx] = region_to_cluster[old_id]
        
    print(f"[SOTA Engine] Optimization complete. Stable Paintable Regions: {remapped_labels.max()}")
    return remapped_labels, final_region_to_cluster, centers_lab

def generate_adobe_outputs(spatial_labels, region_to_cluster, centers_lab, target_w, target_h, font_scale=0.4, text_color=(70,70,70), output_dir="."):
    """
    GLOBAL EDGE ENGINE + LEGEND BUILDER: Generates clean shared boundaries and compiles
    a 3-page comprehensive print document containing the new visual color palette table.
    """
    proc_h, proc_w = spatial_labels.shape[:2]
    hd_scale = min(target_w / proc_w, target_h / proc_h)
    hd_w, hd_h = int(proc_w * hd_scale), int(proc_h * hd_scale)
    
    # 1. Upscale integer label array safely via Nearest Neighbor to protect border positions
    hd_spatial_labels = cv2.resize(spatial_labels, (hd_w, hd_h), interpolation=cv2.INTER_NEAREST)
    
    # 2. Render high-resolution color guide reference map
    hd_color_map = region_to_cluster[hd_spatial_labels]
    hd_color_lab = centers_lab[hd_color_map].reshape(hd_h, hd_w, 3).astype(np.uint8)
    hd_color_rgb = cv2.cvtColor(hd_color_lab, cv2.COLOR_LAB2RGB)
    hd_color_bgr = cv2.cvtColor(hd_color_rgb, cv2.COLOR_RGB2BGR)
    
    # Extract isolated RGB definitions for the palette builder page
    colors_rgb = []
    colors_bgr = []
    for center in centers_lab:
        lab_pixel = np.array([[center]], dtype=np.uint8)
        rgb_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2RGB)[0][0]
        colors_rgb.append((int(rgb_pixel[0]), int(rgb_pixel[1]), int(rgb_pixel[2])))
        colors_bgr.append((int(rgb_pixel[2]), int(rgb_pixel[1]), int(rgb_pixel[0])))
    
    # 3. GLOBAL EDGE DERIVATION: Generates perfectly shared single-pixel boundary paths
    edges = np.zeros((hd_h, hd_w), dtype=np.uint8)
    edges[:, :-1] |= (hd_spatial_labels[:, :-1] != hd_spatial_labels[:, 1:])
    edges[:-1, :] |= (hd_spatial_labels[:-1, :] != hd_spatial_labels[1:, :])
    
    edges[0, :] = 1; edges[-1, :] = 1; edges[:, 0] = 1; edges[:, -1] = 1
    
    hd_line_target = np.ones((hd_h, hd_w, 3), dtype=np.uint8) * 255
    hd_line_target[edges > 0] = (50, 50, 50)
    
    # 4. Place number labels inside canvas regions
    unique_regions = np.unique(spatial_labels)
    for r in unique_regions:
        if r == 0: continue
        ys, xs = np.where(spatial_labels == r)
        if len(ys) < 8: continue 
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        
        mask_crop = np.zeros((y_max - y_min + 3, x_max - x_min + 3), dtype=np.uint8)
        mask_crop[ys - y_min + 1, xs - x_min + 1] = 255
        
        dist_map = cv2.distanceTransform(mask_crop, cv2.DIST_L2, 3)
        _, _, _, max_loc = cv2.minMaxLoc(dist_map)
        cx_target = int((x_min - 1 + max_loc[0]) * hd_scale + hd_scale // 2)
        cy_target = int((y_min - 1 + max_loc[1]) * hd_scale + hd_scale // 2)
        
        if len(ys) > 30:
            text = str(region_to_cluster[r])
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, 1)
            text_x = cx_target - text_w // 2
            text_y = cy_target + text_h // 2
            if 0 <= text_x < hd_w and 0 <= text_y < hd_h:
                cv2.putText(hd_line_target, text, (text_x, text_y), font, font_scale, text_color, 1, cv2.LINE_AA)

    # --- PHASE 5: DYNAMIC CODES FOR COLOR PALETTE LEGEND PAGE (PAGE 3) ---
    palette_canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    
    # Draw Page Header Heading
    cv2.putText(palette_canvas, "COLOR PALETTE LEGEND", (150, 220), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (40, 40, 40), 4, cv2.LINE_AA)
    cv2.line(palette_canvas, (150, 260), (target_w - 150, 260), (200, 200, 200), 2, cv2.LINE_AA)
    
    # Define dual-column table layout constraints
    start_y = 400
    row_gap = 340
    col_gap = 1150
    
    for i in range(len(centers_lab)):
        r, g, b = colors_rgb[i]
        bgr = colors_bgr[i]
        hex_code = f"#{r:02X}{g:02X}{b:02X}"
        
        # Determine 8x2 grid coordinate configurations
        col_idx = i // 8
        row_idx = i % 8
        
        x_base = 200 + (col_idx * col_gap)
        y_base = start_y + (row_idx * row_gap)
        
        # Render Visual Color Swatch Box
        cv2.rectangle(palette_canvas, (x_base, y_base), (x_base + 140, y_base + 140), bgr, -1)
        cv2.rectangle(palette_canvas, (x_base, y_base), (x_base + 140, y_base + 140), (60, 60, 60), 2, cv2.LINE_AA) # Swatch outline
        
        # Overlay alphanumeric mappings: [Number ID] -> [Hex String]
        text_str = f"Ink ID: {i}"
        hex_str = f"Hex: {hex_code}"
        
        cv2.putText(palette_canvas, text_str, (x_base + 190, y_base + 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(palette_canvas, hex_str, (x_base + 190, y_base + 115), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (90, 90, 90), 2, cv2.LINE_AA)

    # Padding alignment enclosures
    final_reference = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    final_canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    pad_top = (target_h - hd_h) // 2
    pad_left = (target_w - hd_w) // 2
    
    final_reference[pad_top:pad_top+hd_h, pad_left:pad_left+hd_w] = hd_color_bgr
    final_canvas[pad_top:pad_top+hd_h, pad_left:pad_left+hd_w] = hd_line_target

    ref_path = os.path.join(output_dir, "merged_output_hd_v4.png")
    canvas_path = os.path.join(output_dir, "paint_by_numbers_canvas.png")
    palette_path = os.path.join(output_dir, "paint_by_numbers_palette.png")
    pdf_path = os.path.join(output_dir, "paint_by_numbers_canvas.pdf")  
    
    cv2.imwrite(ref_path, final_reference)
    cv2.imwrite(canvas_path, final_canvas)
    cv2.imwrite(palette_path, palette_canvas)
    
    # --- COMBINE INTO THE 3-PAGE PRODUCTION PACKAGE ---
    page1 = Image.open(ref_path).convert('RGB')
    page2 = Image.open(canvas_path).convert('RGB')
    page3 = Image.open(palette_path).convert('RGB')
    
    page1.save(pdf_path, "PDF", save_all=True, append_images=[page2, page3])
    print("[Success] High-Fidelity 3-Page Production PDF Legend Package generated perfectly.")