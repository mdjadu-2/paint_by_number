import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from streamlit_gsheets import GSheetsConnection
import utils

# --- STYLING & CONFIG ---
st.set_page_config(page_title="AI Paint-by-Numbers", page_icon="🎨", layout="centered")

# Locked Senior-Validated V1 Constants
DPI = 300
TARGET_W = 2480  
TARGET_H = 3508  
SCALE_FACTOR = 4
K = 16
BILATERAL_D = 7
BILATERAL_SIGMA_COLOR = 60
BILATERAL_SIGMA_SPACE = 50
MIN_PAINT_MM = 3.5
CONTRAST_THRESHOLD = 45.0
HARD_FLOOR = 20
FONT_SCALE = 0.4
TEXT_COLOR = (70, 70, 70)

st.title("🎨 Custom Paint-by-Numbers Generator")
st.write("Convert any portrait into a clean, human-paintable A4 vector canvas.")

# Step 1: User Entry Gate
user_name = st.text_input("Enter your name to unlock the generator:", placeholder="Your Name")

# Step 2: File Handling
uploaded_file = st.file_uploader("Upload your source image (JPG/PNG)", type=["jpg", "jpeg", "png"], disabled=not user_name)

if uploaded_file and user_name:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    temp_source_path = "temp_source.jpg"
    cv2.imwrite(temp_source_path, img_bgr)
    h_orig, w_orig = img_bgr.shape[:2]
    
    st.info("🎯 Settings locked for optimal, human-paintable line art extraction.")
    
    if st.button("Generate Printable PDF Canvas", type="primary"):
        with st.status("Processing algorithm pipeline...", expanded=True) as status:
            
            st.write("Initializing canvas and padding...")
            img_proc = utils.initialize_processing_canvas(temp_source_path, TARGET_W, TARGET_H, SCALE_FACTOR)
            
            st.write("Running bilateral smoothing & K-Means color quantization...")
            labels_2d, centers_lab = utils.preprocess_and_quantize(img_proc, K, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
            
            st.write("Segmenting connected-component color regions...")
            region_map, region_to_cluster = utils.form_initial_regions(labels_2d, K)
            
            st.write("Executing morphological merge engine...")
            merged_proc = utils.merge_regions_advanced(region_map, region_to_cluster, centers_lab, DPI, SCALE_FACTOR, MIN_PAINT_MM, CONTRAST_THRESHOLD, HARD_FLOOR)
            
            st.write("Smoothing high-res boundaries...")
            merged_proc = utils.smooth_label_boundaries(merged_proc)
            
            st.write("Generating final printable layers...")
            utils.generate_final_outputs(merged_proc, region_to_cluster, centers_lab, TARGET_W, TARGET_H, SCALE_FACTOR, FONT_SCALE, TEXT_COLOR, ".")
            
            # --- GOOGLE SHEET METRIC LOGGING ---
            st.write("Logging metrics to project ledger...")
            try:
                conn = st.connection("gsheets", type=GSheetsConnection)
                df = conn.read(worksheet="Sheet1")
                
                new_row = pd.DataFrame([{
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "User Name": user_name,
                    "Original Width": w_orig,
                    "Original Height": h_orig,
                    "Status": "SUCCESS"
                }])
                
                updated_df = pd.concat([df, new_row], ignore_index=True)
                conn.update(worksheet="Sheet1", data=updated_df)
            except Exception as e:
                st.warning(f"Metadata ledger bypassed: {e}")
                
            status.update(label="Processing Complete!", state="complete", expanded=False)
        
        st.success("🎉 Version 1 Canvas Generated Successfully!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image("merged_output_hd_v4.png", caption="Color Guide Reference", use_container_width=True)
        with col2:
            st.image("paint_by_numbers_canvas.png", caption="Printable Line Art Outline", use_container_width=True)
            
        canvas_pdf = "paint_by_numbers_canvas.pdf"
        with open(canvas_pdf, "rb") as pdf_file:
            st.download_button(
                label="📥 Download Printable A4 PDF",
                data=pdf_file,
                file_name=f"{user_name}_paint_by_numbers.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
        # Clear engine cache files from cloud container storage
        for f in [temp_source_path, "paint_by_numbers_canvas.png", canvas_pdf, "merged_output_hd_v4.png"]:
            if os.path.exists(f): os.remove(f)