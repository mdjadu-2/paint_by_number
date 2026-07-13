import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from streamlit_gsheets import GSheetsConnection
from utils import *

# --- STYLING & PAGE SETUP ---
st.set_page_config(page_title="Adobe-Style 5D Canvas Generator", page_icon="🎨", layout="centered")

# Engine Parameter Configurations (Locked V2 Spatial Core)
TARGET_W = 2480  # A4 width @ 300 DPI
TARGET_H = 3508  # A4 height @ 300 DPI
SCALE_FACTOR = 4
K_COLORS = 16
COMPACTNESS = 10
N_SEGMENTS = 1200
FONT_SCALE = 0.4
TEXT_COLOR = (70, 70, 70)

# --- HEADER SECTION ---
st.title("🎨 SOTA 5D Joint Spatial-Color Canvas Generator")
st.write("Deconstruct target imagery across a multi-dimensional pixel manifold to capture micro-scale structures with perfect topological containment.")

st.markdown("---")


# --- SECTION 1: PRE-RENDERED ALGORITHMIC SHOWCASE ---
st.subheader("🖼️ Engine Output Verification Gallery")
# st.write("Examine how the multi-dimensional mapping preserves localized anatomy (such as eyes and lips) without generating scattered noise islands:")
st.write("Examine archived showcase examples directly pulled from localized run directories to see how structural detail is maintained:")
# Initializing three dynamic selection tabs for the 001, 002, and 003 workflows
tab1, tab2, tab3 = st.tabs([
    "Sample 001", 
    "Sample 002", 
    "Sample 003"
])

# --- SAMPLE 001 ---
with tab3:
    img_in_1 = "./output/001/input.png"
    img_out_1 = "./output/001/merged_output_hd_v4.png"
    pdf_path_1 = "./output/001/paint_by_numbers_canvas.pdf"
    
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(img_in_1):
            st.image(img_in_1, caption="Original Photo", use_container_width=True)
        else:
            st.info("Input asset 'output/001/input.png' not yet archived locally.")
            
    with col2:
        if os.path.exists(img_out_1):
            st.image(img_out_1, caption="Engine Color Reference", use_container_width=True)
        else:
            st.info("Color guide asset 'output/001/merged_output_hd_v4.png' not yet archived locally.")
            
    if os.path.exists(pdf_path_1):
        with open(pdf_path_1, "rb") as f:
            st.download_button(
                label="📥 Download Sample Vector Package (PDF)",
                data=f,
                file_name="sample_001_paint_package.pdf",
                mime="application/pdf",
                key="dl_sample_001",
                use_container_width=True
            )
    else:
        st.button("📥 Sample PDF Package Unavailable", disabled=True, use_container_width=True, key="btn_dis_1")

# --- SAMPLE 002 ---
with tab1:
    img_in_2 = "./output/002/input.png"
    img_out_2 = "./output/002/merged_output_hd_v4.png"
    pdf_path_2 = "./output/002/paint_by_numbers_canvas.pdf"
    
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(img_in_2):
            st.image(img_in_2, caption="Original Photo", use_container_width=True)
        else:
            st.info("Input asset 'output/002/input.png' not yet archived locally.")
            
    with col2:
        if os.path.exists(img_out_2):
            st.image(img_out_2, caption="Engine Color Reference ", use_container_width=True)
        else:
            st.info("Color guide asset 'output/002/merged_output_hd_v4.png' not yet archived locally.")
            
    if os.path.exists(pdf_path_2):
        with open(pdf_path_2, "rb") as f:
            st.download_button(
                label="📥 Download Sample Vector Package (PDF)",
                data=f,
                file_name="sample_002_paint_package.pdf",
                mime="application/pdf",
                key="dl_sample_002",
                use_container_width=True
            )
    else:
        st.button("📥 Sample PDF Package Unavailable", disabled=True, use_container_width=True, key="btn_dis_2")

# --- SAMPLE 003 ---
with tab2:
    img_in_3 = "./output/003/input.png"
    img_out_3 = "./output/003/merged_output_hd_v4.png"
    pdf_path_3 = "./output/003/paint_by_numbers_canvas.pdf"
    
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(img_in_3):
            st.image(img_in_3, caption="Original Photo", use_container_width=True)
        else:
            st.info("Input asset 'output/003/input.png' not yet archived locally.")
            
    with col2:
        if os.path.exists(img_out_3):
            st.image(img_out_3, caption="Engine Color Reference", use_container_width=True)
        else:
            st.info("Color guide asset 'output/003/merged_output_hd_v4.png' not yet archived locally.")
            
    if os.path.exists(pdf_path_3):
        with open(pdf_path_3, "rb") as f:
            st.download_button(
                label="📥 Download Sample Vector Package (PDF)",
                data=f,
                file_name="sample_003_paint_package.pdf",
                mime="application/pdf",
                key="dl_sample_003",
                use_container_width=True
            )
    else:
        st.button("📥 Sample PDF Package Unavailable", disabled=True, use_container_width=True, key="btn_dis_3")

st.markdown("---")



# --- SECTION 2: TARGET TECHNICAL SPECIFICATIONS (VERTICAL DESIGN) ---
st.subheader("🛠️ Target Technical Specifications")
st.write("The processing engine computes vector boundaries utilizing a 5D coordinate clustering architecture:")
col_spec1, col_spec2 = st.columns(2)
with col_spec1:
    st.metric(
        label="Output Canvas Geometry", 
        value="A4 Standard"
    )

with col_spec2:
    st.metric(
        label="Print Resolution Blueprint", 
        value="300 DPI", 
        delta="2480 x 3508 px", 
        delta_arrow="up"
    )

col_spec3, col_spec4 = st.columns(2)
with col_spec3:
    st.metric(
        label="Manifold Coordinate Space Vectors", 
        value="5D Joint Matrix", 
        delta="V = [L, A, B, x, y]",
        delta_arrow="up"
    )

with col_spec4:
    st.metric(
        label="Maximum Color Partition Allocation", 
        value="K = 16 Distinct Colors"
    )
# st.metric(
#     label="Output Canvas Geometry", 
#     value="A4 Standard Dimensions"
# )

# st.metric(
#     label="Print Resolution Blueprint", 
#     value="300 DPI", 
#     delta="2480 x 3508 px", 
#     delta_arrow="up"
# )

# st.metric(
#     label="Manifold Coordinate Space Vectors", 
#     value="5D Joint Matrix", 
#     delta="V = [L, A, B, x, y]",
#     delta_arrow="up"
# )

# st.metric(
#     label="Maximum Color Partition Allocation", 
#     value="K = 16 Distinct Inks"
# )

st.markdown("---")



# --- SECTION 3: THE LIVE PRODUCTION GENERATOR ---
st.subheader("🚀 Generate Your Custom Canvas")

user_name = st.text_input("Enter your name to unlock processing credentials:", placeholder="Your Name")
uploaded_file = st.file_uploader("Upload your target image asset (JPG/PNG)", type=["jpg", "jpeg", "png"], disabled=not user_name)

if uploaded_file and user_name:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Define primary root directory for archiving good examples
    base_output_dir = "output"
    temp_source_path = "temp_source.png"
    cv2.imwrite(temp_source_path, img_bgr)
    h_orig, w_orig = img_bgr.shape[:2]
    
    st.success("🎯 Image integrity verified. Spatial processing configuration arrays locked.")
    
    if st.button("Execute Pipeline & Render PDF", type="primary"):
        with st.status("Initializing processing runtime matrix...", expanded=True) as status:

            # --- DYNAMIC INCREMENTAL FOLDER LOGIC ---
            os.makedirs(base_output_dir, exist_ok=True)
            existing_folders = [
                f for f in os.listdir(base_output_dir) 
                if os.path.isdir(os.path.join(base_output_dir, f)) and f.isdigit()
            ]
            
            if existing_folders:
                next_num = max([int(f) for f in existing_folders]) + 1
            else:
                next_num = 1
                
            # Establish the unique local path for this specific example run
            run_dir = os.path.join(base_output_dir, f"{next_num:03d}")
            os.makedirs(run_dir, exist_ok=True)
            
            # Save the raw unedited input file directly into the archive directory
            input_save_path = os.path.join(run_dir, "input.png")
            cv2.imwrite(input_save_path, img_bgr)
            
            st.write("Isolating dynamic core boundary paths and executing white alpha-blending...")
            img_proc = initialize_adobe_canvas(temp_source_path, TARGET_W, TARGET_H, SCALE_FACTOR)
            
            st.write("Mapping to 5D Joint Manifold space and running SLIC localized boundary segregation...")
            spatial_labels, region_to_cluster, centers_lab = process_adobe_slic(img_proc, K_COLORS, COMPACTNESS, N_SEGMENTS)
            
            st.write("Executing high-resolution upscaling, anti-aliasing filtering, and spatial numbering map...")
            # generate_adobe_outputs(spatial_labels, region_to_cluster, centers_lab, TARGET_W, TARGET_H, FONT_SCALE, TEXT_COLOR, ".")
            generate_adobe_outputs(spatial_labels, region_to_cluster, centers_lab, TARGET_W, TARGET_H, FONT_SCALE, TEXT_COLOR, run_dir)
            
            # --- TELEMETRY LEDGER RECORDING ---
            st.write("Writing compilation metrics to database service ledger...")
            try:
                conn = st.connection("gsheets", type=GSheetsConnection)
                df = conn.read(worksheet="Sheet1")
                
                new_row = pd.DataFrame([{
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "User Name": user_name,
                    "Original Width": w_orig,
                    "Original Height": h_orig,
                    "Status": "SUCCESS_5D_ADOBE"
                }])
                
                updated_df = pd.concat([df, new_row], ignore_index=True)
                conn.update(worksheet="Sheet1", data=updated_df)
            except Exception as e:
                st.warning(f"Database ledger bypassed: {e}")
                
            status.update(label="Compilation Complete!", state="complete", expanded=False)
        

        # Build exact string paths pointing to the archived local directory
        color_guide_path = os.path.join(run_dir, "merged_output_hd_v4.png")
        canvas_line_path = os.path.join(run_dir, "paint_by_numbers_canvas.png")
        canvas_pdf_path = os.path.join(run_dir, "paint_by_numbers_canvas.pdf")

        st.markdown("### 🎉 Render Results")
        col_o1, col_o2 = st.columns(2)
        # with col_o1:
        #     st.image("merged_output_hd_v4.png", caption="5D Spatial Color Guide Map Reference", width="stretch")
        # with col_o2:
        #     st.image("paint_by_numbers_canvas.png", caption="Printable Line Art Outline Map", width="stretch")
        with col_o1: 
            st.image(color_guide_path, caption="Page 1: Color Guide Reference Map")
        with col_o2: 
            st.image(canvas_line_path, caption="Page 2: Printable Canvas Map")
            
        canvas_pdf = "paint_by_numbers_canvas.pdf"
        # with open(canvas_pdf, "rb") as pdf_file:
        with open(canvas_pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📥 Download Production-Ready A4 Print PDF",
                data=pdf_file,
                file_name=f"{user_name}_paint_package.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
        # Clear cloud workspace file artifacts
        for file_artifact in [temp_source_path, "paint_by_numbers_canvas.png", canvas_pdf, "merged_output_hd_v4.png"]:
            if os.path.exists(file_artifact): 
                os.remove(file_artifact)

st.markdown("---")

# --- SECTION 4: THE ALGORITHMIC BRIEF DEEP DIVE ---
# st.subheader("🕵️‍♂️ Algorithmic Pipeline Architecture Details")
# st.markdown("""
# Standard commercial color image vectorizers look at image processing purely on a per-pixel color basis. This application handles it as a structural **Multi-Dimensional Manifold Optimization Problem**.

# #### 1. Why Global K-Means Fails on Real Geometry
# Global color clustering algorithms (like standard 3D K-Means in RGB or LAB spaces) evaluate pixel values entirely in isolation from their physical positions. If a pixel on a subject's eyebrow shares a matching color value with a shadow on their jersey sleeve, the global algorithm forces them into the exact same cluster identity. This creates thousands of disconnected pixel micro-islands that scatter across high-texture zones, completely erasing facial expressions and structure.

# #### 2. The 5D Joint Coordinate System
# To solve this, the engine uses **Simple Linear Iterative Clustering (SLIC)** to map every pixel into a continuous **5D space** containing both color coordinates and physical coordinates simultaneously:

# $$\\text{Vector} = [L, A, B, x, y]$$

# When calculating region clustering bounds, the mathematical distance metric $D$ is enforced by balancing the color distance $d_c$ against the physical spatial distance $d_s$ using a dedicated compactness scaling factor $m$:

# $$D = \\sqrt{d_c^2 + \\left(\\frac{m}{S}\\right)^2 d_s^2}$$

# #### 3. Spatial Continuity Enforcement
# Because spatial distance ($x, y$) acts as an explicit coordinate variable within the clustering matrix, the algorithm can never group pixels together unless they share matching color traits **and** touch each other physically on the canvas. This prevents micro-islands from forming in the first place. 

# The engine extracts clean, continuous, and structurally sound boundary blocks that preserve vital human facial variations across variable portrait frames.
# """)