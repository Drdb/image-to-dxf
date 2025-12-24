"""
Image to DXF Converter - Web Application
A simple web service for converting bitmap images to DXF format
"""

import streamlit as st
from PIL import Image
import io
from converter import BitmapToDXFConverter

# Page configuration
st.set_page_config(
    page_title="Image to DXF Converter",
    page_icon="☕",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .coffee-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
    }
    .coffee-price {
        font-size: 1.8rem;
        font-weight: bold;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Payment links (Stripe - LIVE)
PAYMENT_LINKS = {
    "deli_coffee": "https://buy.stripe.com/4gM7sK7524Lrarof3Ibo400",
    "cappuccino": "https://buy.stripe.com/5kQ6oG896b9PdDAcVAbo401",
    "specialty_espresso": "https://buy.stripe.com/dRm28qcpmgu9fLI6xcbo402"
}

# Initialize converter
converter = BitmapToDXFConverter()

# Header
st.markdown("<h1 class='main-header'>☕ Image to DXF Converter</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Convert your bitmap images to DXF format for laser cutting and CNC</p>", unsafe_allow_html=True)

st.divider()

# File upload section
st.subheader("📁 Step 1: Upload Your Image")
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["png", "jpg", "jpeg", "bmp", "gif", "tiff"],
    help="Supported formats: PNG, JPG, JPEG, BMP, GIF, TIFF"
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
    
    with col2:
        st.markdown("**Image Details:**")
        st.write(f"• Size: {image.size[0]} × {image.size[1]} pixels")
        st.write(f"• Format: {image.format or 'Unknown'}")
        st.write(f"• Mode: {image.mode}")
    
    st.divider()
    
    # Conversion settings
    st.subheader("⚙️ Step 2: Configure Conversion")
    
    col1, col2 = st.columns(2)
    
    with col1:
        mode = st.selectbox(
            "Conversion Mode",
            options=["threshold", "floyd_steinberg", "outline"],
            format_func=lambda x: {
                "threshold": "Threshold (Lines)",
                "floyd_steinberg": "Floyd-Steinberg Dithering (Dots)",
                "outline": "Outline (Contours)"
            }[x],
            help="Choose how the image should be converted"
        )
        
        image_height = st.number_input(
            "Output Height (microns)",
            min_value=100.0,
            max_value=100000.0,
            value=1000.0,
            step=100.0,
            help="The height of the output in microns"
        )
        
        spot_size = st.number_input(
            "Laser Spot Size (microns)",
            min_value=1.0,
            max_value=100.0,
            value=5.0,
            step=1.0,
            help="Your laser's spot size"
        )
    
    with col2:
        if mode == "threshold":
            threshold = st.slider(
                "Threshold",
                min_value=0,
                max_value=255,
                value=200,
                help="Pixels darker than this become black"
            )
        else:
            threshold = 200
        
        invert = st.checkbox("Invert (swap black/white)", value=False)
        flip_y = st.checkbox("Flip Y axis (CAD orientation)", value=True)
        bidirectional = st.checkbox("Bidirectional scan (zigzag)", value=True)
        
        if mode == "outline":
            outline_levels = st.slider("Contour Levels", 2, 16, 2)
            smoothing = st.slider("Smoothing", 0.0, 10.0, 2.0)
        else:
            outline_levels = 2
            smoothing = 2.0
    
    st.divider()
    
    # Payment section
    st.subheader("☕ Step 3: Buy the Developer a Coffee")
    st.markdown("Choose based on your image complexity:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="coffee-card">
            <h3>🥤 Deli Coffee</h3>
            <p class="coffee-price">$2.50</p>
            <p>Simple graphics<br>Basic shapes</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Buy Deli Coffee ☕", key="deli"):
            st.markdown(f"[Click here to pay →]({PAYMENT_LINKS['deli_coffee']})")
    
    with col2:
        st.markdown("""
        <div class="coffee-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h3>☕ Cappuccino</h3>
            <p class="coffee-price">$5.00</p>
            <p>Average complexity<br>Detailed images</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Buy Cappuccino ☕", key="cappuccino"):
            st.markdown(f"[Click here to pay →]({PAYMENT_LINKS['cappuccino']})")
    
    with col3:
        st.markdown("""
        <div class="coffee-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h3>✨ Specialty Espresso</h3>
            <p class="coffee-price">$6.50</p>
            <p>Complex files<br>High detail work</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Buy Specialty Espresso ☕", key="specialty"):
            st.markdown(f"[Click here to pay →]({PAYMENT_LINKS['specialty_espresso']})")
    
    st.info("💡 **Tip:** Payment is on the honor system. Click a button above to pay, then convert your file below.")
    
    st.divider()
    
    # Convert button
    st.subheader("🔄 Step 4: Convert & Download")
    
    if st.button("🚀 Convert to DXF", type="primary", use_container_width=True):
        with st.spinner("Converting... Please wait."):
            try:
                # Reset file position
                uploaded_file.seek(0)
                image_bytes = uploaded_file.read()
                
                # Convert
                dxf_content, stats = converter.convert(
                    input_image=image_bytes,
                    mode=mode,
                    image_height_um=image_height,
                    spot_size_um=spot_size,
                    spot_spacing_factor=1.1,
                    threshold=threshold,
                    invert=invert,
                    flip_y=flip_y,
                    bidirectional=bidirectional,
                    outline_levels=outline_levels,
                    smoothing_amount=smoothing,
                    corner_threshold=45.0
                )
                
                # Store in session state
                st.session_state['dxf_content'] = dxf_content
                st.session_state['stats'] = stats
                st.session_state['filename'] = uploaded_file.name.rsplit('.', 1)[0] + '.dxf'
                
                st.success("✅ Conversion successful!")
                
            except Exception as e:
                st.error(f"❌ Conversion failed: {str(e)}")
    
    # Download section
    if 'dxf_content' in st.session_state:
        stats = st.session_state['stats']
        
        st.markdown("### 📊 Conversion Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Entities", f"{stats['entity_count']:,}")
        with col2:
            st.metric("Output Size", f"{stats['width_um']:.0f} × {stats['height_um']:.0f} µm")
        with col3:
            st.metric("File Size", f"{stats['file_size_kb']:.1f} KB")
        
        mode_names = {
            "threshold": "Threshold (Lines)",
            "floyd_steinberg": "Floyd-Steinberg (Dots)",
            "outline": "Outline (Contours)"
        }
        st.write(f"**Mode:** {mode_names.get(stats['mode'], stats['mode'])}")
        
        # Download button
        st.download_button(
            label="⬇️ Download DXF File",
            data=st.session_state['dxf_content'],
            file_name=st.session_state['filename'],
            mime="application/dxf",
            use_container_width=True
        )
        
        st.balloons()

else:
    # No file uploaded yet
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 10px; margin: 2rem 0;">
        <h3>👆 Upload an image to get started</h3>
        <p style="color: #666;">Supported formats: PNG, JPG, BMP, GIF, TIFF</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9rem;">
    <p>Image to DXF Converter | Built with ❤️ for the maker community</p>
    <p>Questions? Contact: <a href="mailto:support@example.com">support@example.com</a></p>
</div>
""", unsafe_allow_html=True)
    
