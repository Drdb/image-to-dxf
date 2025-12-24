# Image to DXF Converter

A web application for converting bitmap images to DXF format for laser cutting and CNC.

## Files

- `app.py` - The main Streamlit web application
- `converter.py` - The core conversion engine
- `requirements.txt` - Python dependencies

## Features

- **Three conversion modes:**
  - Threshold (Lines) - for simple black/white graphics
  - Floyd-Steinberg Dithering (Dots) - for photos and grayscale
  - Outline (Contours) - for vector-style output

- **Configurable parameters:**
  - Output height in microns
  - Laser spot size
  - Threshold level
  - Invert colors
  - Flip Y axis
  - Bidirectional scanning

## Payment Tiers

- ☕ Deli Coffee ($2.50) - Simple graphics
- ☕ Cappuccino ($5.00) - Average complexity
- ✨ Specialty Espresso ($6.50) - Complex files

## Deployment

See step-by-step instructions for deploying to Streamlit Cloud.
