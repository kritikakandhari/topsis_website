# Topsis Image Downloader Web Service

A professional web application built with Flask that allows users to search for images, automatically ranks them using the **Topsis Algorithm** based on resolution and file size, and delivers a curated ZIP file via email.

## 🚀 Live Demo
**[https://topsis-websitee.vercel.app](https://topsis-websitee.vercel.app)**

## Features
- **Deterministic Search**: Ensures fair usage and unique results.
- **Topsis Ranking**: Curates the best images based on Width, Height, and File Size.
- **Dual Tiers**:
  - **Free**: Up to 50 images with ad support.
  - **Paid**: Unlimited images (₹0.50/extra) with real-time UPI payment integration.
- **Secure Payments**: Integrated UPI QR Code generation with UTR verification.
- **Fast Delivery**: Concurrent image processing and email delivery.

## Tech Stack
- **Backend**: Python, Flask, Gunicorn
- **Data Analysis**: Pandas, NumPy (Topsis Implementation)
- **Frontend**: HTML5, Vanilla CSS (Glassmorphism), QRious (QR Code)
- **Deployment**: Vercel Serverless

## Configuration
To run your own instance, set these environment variables in Vercel:
- `UPI_ID`: Your payment VPA (e.g., `user@upi`)
- `SENDER_EMAIL`: Gmail account for sending images.
- `SENDER_PASSWORD`: Gmail App Password.

---
Developed as part of the TOPSIS Data Analysis Assignment.
