"""
Image to DXF Converter - Web Application
A beautifully designed web service for converting bitmap images to DXF format
"""

import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageEnhance
import io
import base64
import math
from converter import BitmapToDXFConverter

# Page configuration
st.set_page_config(
    page_title="Image to DXF Converter",
    page_icon="☕",
    layout="wide"
)

# Beautiful custom CSS with fixed input text colors
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+Pro:wght@300;400;600&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }
    
    /* Typography */
    h1, h2, h3 {
        font-family: 'Playfair Display', Georgia, serif !important;
        color: #f8f0e3 !important;
    }
    
    p, li, span, label, .stMarkdown {
        font-family: 'Source Sans Pro', sans-serif !important;
    }
    
    /* Make all form labels legible */
    .stSelectbox label, .stNumberInput label, .stSlider label, 
    .stCheckbox label, .stRadio label, .stTextInput label,
    .stFileUploader label {
        color: #e8d5b5 !important;
        font-weight: 500 !important;
    }
    
    /* Make checkbox and radio text visible */
    .stCheckbox span, .stRadio span, .stCheckbox p, .stRadio p,
    .stRadio label span, .stRadio div[data-testid="stMarkdownContainer"] p {
        color: #e8d5b5 !important;
    }
    
    /* FIXED: Input field text - BLACK on white background */
    .stNumberInput input, .stTextInput input {
        background: #ffffff !important;
        border: 1px solid rgba(232,180,120,0.5) !important;
        color: #1a1a2e !important;
        border-radius: 6px !important;
    }
    
    .stNumberInput input:focus, .stTextInput input:focus {
        border-color: #e8b478 !important;
        box-shadow: 0 0 0 2px rgba(232,180,120,0.2) !important;
    }
    
    /* Hero header */
    .hero-container {
        text-align: center;
        padding: 1.5rem 1rem;
        margin-bottom: 1.5rem;
        background: linear-gradient(180deg, rgba(232,180,120,0.1) 0%, rgba(0,0,0,0) 100%);
        border-radius: 20px;
    }
    
    .hero-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: #e8b478;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: 2px;
    }
    
    .hero-subtitle {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 1.1rem;
        color: #a0a0b0;
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    /* Section titles */
    .section-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.2rem;
        color: #e8b478;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Cards */
    .custom-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(232,180,120,0.2);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Image container - smaller, with visible border for white backgrounds */
    .image-container {
        background: rgba(0,0,0,0.3);
        border: 2px solid rgba(232,180,120,0.4);
        border-radius: 8px;
        padding: 0.5rem;
        text-align: center;
    }
    
    .image-label {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.75rem;
        color: #e8b478;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(232,180,120,0.2);
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .stRadio > div > div {
        gap: 0.3rem !important;
    }
    
    /* Coffee tier cards */
    .coffee-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.8rem;
        margin: 1rem 0;
    }
    
    .coffee-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(232,180,120,0.3);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .coffee-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #e8b478, #d4a574, #c9956c);
    }
    
    .coffee-card:hover {
        transform: translateY(-3px);
        border-color: #e8b478;
        box-shadow: 0 8px 30px rgba(232,180,120,0.2);
    }
    
    .coffee-icon {
        font-size: 1.8rem;
        margin-bottom: 0.3rem;
    }
    
    .coffee-name {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.1rem;
        color: #f8f0e3;
        margin-bottom: 0.2rem;
    }
    
    .coffee-price {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #e8b478;
        margin: 0.3rem 0;
    }
    
    .coffee-desc {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.8rem;
        color: #a0a0b0;
        line-height: 1.3;
    }
    
    /* Stats display */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .stat-box {
        background: rgba(232,180,120,0.1);
        border: 1px solid rgba(232,180,120,0.2);
        border-radius: 8px;
        padding: 0.6rem;
        text-align: center;
    }
    
    .stat-value {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.2rem;
        color: #e8b478;
        font-weight: 600;
    }
    
    .stat-label {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.7rem;
        color: #a0a0b0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Unit info box */
    .unit-info {
        background: rgba(232,180,120,0.08);
        border-left: 3px solid #e8b478;
        padding: 0.6rem 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #c8b8a0;
    }
    
    /* Buttons */
    .stButton > button {
        font-family: 'Source Sans Pro', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 20px rgba(232,180,120,0.3) !important;
    }
    
    /* Download button special styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #e8b478 0%, #d4a574 100%) !important;
        color: #1a1a2e !important;
        font-weight: 600 !important;
        border: none !important;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background: rgba(255,255,255,0.05) !important;
        border: 2px dashed rgba(232,180,120,0.4) !important;
        border-radius: 8px !important;
    }
    
    .stFileUploader label {
        color: #e8d5b5 !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: #ffffff !important;
        border-color: rgba(232,180,120,0.5) !important;
        color: #1a1a2e !important;
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background: #e8b478 !important;
    }
    
    .stSlider label {
        color: #e8d5b5 !important;
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(232,180,120,0.1) !important;
        border: 1px solid rgba(232,180,120,0.3) !important;
        border-radius: 8px !important;
        color: #e8d5b5 !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(232,180,120,0.2) !important;
        margin: 1.5rem 0 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        border-top: 1px solid rgba(232,180,120,0.2);
    }
    
    .footer-text {
        font-family: 'Source Sans Pro', sans-serif;
        color: #606070;
        font-size: 0.85rem;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .hero-title { font-size: 2rem; }
        .coffee-grid { grid-template-columns: 1fr; }
        .stats-grid { grid-template-columns: repeat(2, 1fr); }
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


def generate_preview(image, mode, threshold, invert, line_step, brightness=1.0, contrast=1.0):
    """Generate a high-fidelity preview of the DXF output - black on white like CAD software."""
    img = image.copy().convert('L')
    w, h = img.size
    
    # Apply brightness and contrast adjustments for dithering mode
    if mode == "floyd_steinberg":
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)
    
    # Create preview image - WHITE background, BLACK lines (like CAD software)
    preview = Image.new('RGB', (w, h), '#ffffff')
    draw = ImageDraw.Draw(preview)
    
    # Apply inversion to source if needed
    if invert:
        img = ImageOps.invert(img)
    
    px = img.load()
    
    if mode == "floyd_steinberg":
        # Apply actual Floyd-Steinberg dithering
        pixels = list(img.getdata())
        data = [[float(pixels[y * w + x]) for x in range(w)] for y in range(h)]
        
        for y in range(h):
            for x in range(w):
                old_pixel = data[y][x]
                new_pixel = 255 if old_pixel > 127 else 0
                data[y][x] = new_pixel
                error = old_pixel - new_pixel
                
                if x + 1 < w:
                    data[y][x + 1] += error * 7 / 16
                if y + 1 < h:
                    if x > 0:
                        data[y + 1][x - 1] += error * 3 / 16
                    data[y + 1][x] += error * 5 / 16
                    if x + 1 < w:
                        data[y + 1][x + 1] += error * 1 / 16
        
        # Draw small BLACK dots at actual line_step intervals
        dot_radius = max(0, min(1, line_step // 4))
        for y in range(0, h, line_step):
            for x in range(0, w, line_step):
                if data[y][x] < 128:
                    if dot_radius == 0:
                        draw.point((x, y), fill='#000000')
                    else:
                        draw.ellipse(
                            [x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius],
                            fill='#000000'
                        )
    
    elif mode == "outline":
        # Marching squares edge detection for accurate preview
        edges_img = img.filter(ImageFilter.FIND_EDGES)
        edges_img = edges_img.filter(ImageFilter.SMOOTH)
        edges_px = edges_img.load()
        
        # Draw thin BLACK contour lines
        for y in range(h):
            for x in range(w):
                if edges_px[x, y] > 40:
                    draw.point((x, y), fill='#000000')
    
    else:  # threshold mode
        # Draw thin BLACK horizontal scan lines exactly as DXF will have them
        for y in range(0, h, line_step):
            x = 0
            while x < w:
                # Find start of black region
                while x < w and px[x, y] >= threshold:
                    x += 1
                if x >= w:
                    break
                x1 = x
                # Find end of black region
                while x < w and px[x, y] < threshold:
                    x += 1
                x2 = x
                # Draw thin BLACK line segment (width=1)
                if x2 > x1:
                    draw.line([(x1, y), (x2, y)], fill='#000000', width=1)
    
    return preview


# ============== MAIN APP ==============

# Hero header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">☕ Image to DXF</div>
    <div class="hero-subtitle">Transform your images into precision laser-ready DXF files</div>
</div>
""", unsafe_allow_html=True)

# Main layout: Settings on left, Images on right
col_settings, col_images = st.columns([1, 2], gap="large")

with col_settings:
    # Upload section
    st.markdown('<div class="section-title">📁 Upload Image</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Drag & drop or click to browse",
        type=["png", "jpg", "jpeg", "bmp", "gif", "tiff"],
        help="Supported formats: PNG, JPG, BMP, GIF, TIFF"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.markdown(f'<div style="color: #e8b478; font-size: 0.85rem;">✓ Loaded: {image.size[0]} × {image.size[1]} px</div>', unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
    
    # Settings section
    st.markdown('<div class="section-title">⚙️ Settings</div>', unsafe_allow_html=True)
    
    # Radio buttons for conversion mode (all 3 visible)
    mode = st.radio(
        "Conversion Mode",
        options=["threshold", "floyd_steinberg", "outline"],
        format_func=lambda x: {
            "threshold": "🔲 Threshold (Lines)",
            "floyd_steinberg": "⚫ Dithering (Dots)",
            "outline": "✏️ Outline (Contours)"
        }[x],
        horizontal=False
    )
    
    st.markdown('<div class="section-title" style="margin-top: 1rem;">📐 Dimensions</div>', unsafe_allow_html=True)
    
    # Unit explanation
    st.markdown("""
    <div class="unit-info">
        💡 <strong>Drawing Units</strong> are scalable — enter values as µm, mm, inches, or any unit.
    </div>
    """, unsafe_allow_html=True)
    
    output_height = st.number_input(
        "Output Height",
        min_value=1.0,
        max_value=1000000.0,
        value=1000.0,
        step=100.0,
        help="Height in your chosen drawing units"
    )
    
    spot_size = st.number_input(
        "Tool / Laser Spot Size",
        min_value=0.1,
        max_value=1000.0,
        value=5.0,
        step=1.0,
        help="Tool or laser spot size in same units"
    )
    
    # Mode-specific settings
    if mode == "threshold":
        threshold = st.slider("Threshold", 0, 255, 200, help="Pixels darker than this become marks")
        brightness = 1.0
        contrast = 1.0
    elif mode == "floyd_steinberg":
        threshold = 200
        st.markdown('<div style="color: #e8d5b5; font-size: 0.9rem; margin-top: 0.5rem;">Image Adjustments</div>', unsafe_allow_html=True)
        brightness = st.slider("Brightness", 0.2, 2.0, 1.0, 0.1, help="Adjust image brightness before dithering")
        contrast = st.slider("Contrast", 0.2, 2.0, 1.0, 0.1, help="Adjust image contrast before dithering")
    else:
        threshold = 200
        brightness = 1.0
        contrast = 1.0
    
    if mode == "outline":
        outline_levels = st.slider("Contour Levels", 2, 16, 2)
        smoothing = st.slider("Smoothing", 0.0, 10.0, 2.0)
    else:
        outline_levels = 2
        smoothing = 2.0
    
    # Options
    st.markdown('<div style="color: #e8d5b5; font-size: 0.9rem; margin-top: 0.5rem;">Options</div>', unsafe_allow_html=True)
    col_o1, col_o2 = st.columns(2)
    with col_o1:
        invert = st.checkbox("Invert", value=False)
        flip_y = st.checkbox("Flip Y", value=True)
    with col_o2:
        bidirectional = st.checkbox("Bidirectional", value=True)

with col_images:
    if uploaded_file:
        # Calculate parameters for preview
        w, h = image.size
        aspect_ratio = w / h
        output_width = output_height * aspect_ratio
        pixel_size_y = output_height / h
        min_spacing = spot_size * 1.1
        line_step = max(1, int(math.ceil(min_spacing / pixel_size_y)))
        
        # Two images side by side
        col_orig, col_preview = st.columns(2, gap="small")
        
        with col_orig:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.markdown('<div class="image-label">Original Image</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_preview:
            st.markdown('<div class="image-container" style="background: #ffffff;">', unsafe_allow_html=True)
            st.markdown('<div class="image-label" style="color: #1a1a2e;">DXF Preview</div>', unsafe_allow_html=True)
            preview_img = generate_preview(image, mode, threshold, invert, line_step, brightness, contrast)
            st.image(preview_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Stats below both images
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value">{output_width:.0f}</div>
                <div class="stat-label">Width</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{output_height:.0f}</div>
                <div class="stat-label">Height</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{line_step}</div>
                <div class="stat-label">Step</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{h // line_step}</div>
                <div class="stat-label">Lines</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Example images section
        st.markdown("""
        <div style="margin-top: 1.5rem;">
            <div class="section-title" style="font-size: 1rem;">📋 Complexity Examples</div>
        </div>
        """, unsafe_allow_html=True)
        
        col_ex1, col_ex2, col_ex3 = st.columns(3)
        
        # Load example images safely - won't crash if files are missing
        import os
        example_path = os.path.join(os.path.dirname(__file__), "examples")
        
        with col_ex1:
            st.markdown('<div style="text-align: center; color: #e8b478; font-size: 0.8rem; margin-bottom: 0.3rem;">🥤 Simple (Outline)</div>', unsafe_allow_html=True)
            simple_path = os.path.join(example_path, "simple_outline.png")
            if os.path.exists(simple_path):
                st.image(simple_path, use_container_width=True)
            else:
                st.markdown('<div style="background: #2a2a3e; padding: 2rem; text-align: center; border-radius: 8px; color: #666;">Example image</div>', unsafe_allow_html=True)
        
        with col_ex2:
            st.markdown('<div style="text-align: center; color: #e8b478; font-size: 0.8rem; margin-bottom: 0.3rem;">☕ Medium (Threshold)</div>', unsafe_allow_html=True)
            medium_path = os.path.join(example_path, "medium_threshold.png")
            if os.path.exists(medium_path):
                st.image(medium_path, use_container_width=True)
            else:
                st.markdown('<div style="background: #2a2a3e; padding: 2rem; text-align: center; border-radius: 8px; color: #666;">Example image</div>', unsafe_allow_html=True)
        
        with col_ex3:
            st.markdown('<div style="text-align: center; color: #e8b478; font-size: 0.8rem; margin-bottom: 0.3rem;">✨ Complex (Dithering)</div>', unsafe_allow_html=True)
            complex_path = os.path.join(example_path, "complex_dithering.png")
            if os.path.exists(complex_path):
                st.image(complex_path, use_container_width=True)
            else:
                st.markdown('<div style="background: #2a2a3e; padding: 2rem; text-align: center; border-radius: 8px; color: #666;">Example image</div>', unsafe_allow_html=True)
    else:
        # Placeholder when no image
        st.markdown("""
        <div style="display: flex; gap: 1rem; margin-top: 1rem;">
            <div class="image-container" style="flex: 1; min-height: 250px; display: flex; align-items: center; justify-content: center;">
                <div style="text-align: center; color: #606070;">
                    <div style="font-size: 2rem;">🖼️</div>
                    <div style="font-size: 0.85rem;">Original</div>
                </div>
            </div>
            <div class="image-container" style="flex: 1; min-height: 250px; display: flex; align-items: center; justify-content: center; background: #f5f5f5;">
                <div style="text-align: center; color: #909090;">
                    <div style="font-size: 2rem;">👁️</div>
                    <div style="font-size: 0.85rem;">Preview</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Only show payment and convert sections if image is uploaded
if uploaded_file:
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Payment section - more compact
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <div class="section-title" style="justify-content: center; font-size: 1.3rem;">☕ Support the Developer</div>
        <p style="color: #a0a0b0; font-size: 0.9rem; margin: 0;">Choose a coffee based on your image complexity</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="coffee-grid">
        <div class="coffee-card">
            <div class="coffee-icon">🥤</div>
            <div class="coffee-name">Diner Coffee</div>
            <div class="coffee-price">$2.50</div>
            <div class="coffee-desc">Simple graphics & shapes</div>
        </div>
        <div class="coffee-card">
            <div class="coffee-icon">☕</div>
            <div class="coffee-name">Cappuccino</div>
            <div class="coffee-price">$5.00</div>
            <div class="coffee-desc">Average complexity</div>
        </div>
        <div class="coffee-card">
            <div class="coffee-icon">✨</div>
            <div class="coffee-name">Specialty Espresso</div>
            <div class="coffee-price">$6.50</div>
            <div class="coffee-desc">Complex & detailed</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.link_button("Buy Diner Coffee", PAYMENT_LINKS["deli_coffee"], use_container_width=True)
    with col_p2:
        st.link_button("Buy Cappuccino", PAYMENT_LINKS["cappuccino"], use_container_width=True)
    with col_p3:
        st.link_button("Buy Specialty", PAYMENT_LINKS["specialty_espresso"], use_container_width=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Convert section
    col_c1, col_c2, col_c3 = st.columns([1, 2, 1])
    with col_c2:
        convert_clicked = st.button("⚡ Convert to DXF", type="primary", use_container_width=True)
    
    if convert_clicked:
        with st.spinner("Converting..."):
            try:
                uploaded_file.seek(0)
                
                # For dithering mode, apply brightness/contrast to image before conversion
                if mode == "floyd_steinberg" and (brightness != 1.0 or contrast != 1.0):
                    img_for_conversion = image.copy().convert('L')
                    if brightness != 1.0:
                        enhancer = ImageEnhance.Brightness(img_for_conversion)
                        img_for_conversion = enhancer.enhance(brightness)
                    if contrast != 1.0:
                        enhancer = ImageEnhance.Contrast(img_for_conversion)
                        img_for_conversion = enhancer.enhance(contrast)
                    image_input = img_for_conversion
                else:
                    uploaded_file.seek(0)
                    image_input = uploaded_file.read()
                
                dxf_content, stats = converter.convert(
                    input_image=image_input,
                    mode=mode,
                    image_height_um=output_height,
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
                
                st.session_state['dxf_content'] = dxf_content
                st.session_state['stats'] = stats
                st.session_state['filename'] = uploaded_file.name.rsplit('.', 1)[0] + '.dxf'
                
                st.success("✅ Conversion complete!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Download section
    if 'dxf_content' in st.session_state:
        stats = st.session_state['stats']
        
        col_r1, col_r2 = st.columns([2, 1])
        
        with col_r1:
            st.markdown(f"""
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-value">{stats['entity_count']:,}</div>
                    <div class="stat-label">Entities</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{stats['width_um']:.0f}</div>
                    <div class="stat-label">Width</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{stats['height_um']:.0f}</div>
                    <div class="stat-label">Height</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{stats['file_size_kb']:.1f} KB</div>
                    <div class="stat-label">File Size</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_r2:
            st.download_button(
                label="⬇️ Download DXF",
                data=st.session_state['dxf_content'],
                file_name=st.session_state['filename'],
                mime="application/dxf",
                use_container_width=True
            )

# Footer
st.markdown("""
<div class="footer">
    <div class="footer-text">
        Image to DXF Converter • Built with ❤️ for makers<br>
        <span style="font-size: 0.75rem;">Precision tools for laser cutting & CNC</span>
    </div>
</div>
""", unsafe_allow_html=True)
