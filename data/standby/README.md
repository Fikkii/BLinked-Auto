# 📸 Standby Personal Photo Directory

This directory holds your personal standby reference photo for AI image generation with **Qwen-Image-3.0**.

### 🌟 How it Works
1. Place your personal reference photo here as `personal_photo.jpg` (or `.png`, `.jpeg`, `.webp`).
2. Whenever a LinkedIn post is generated, our pipeline:
   - Uses your post context, topic, and tutoring/engineering story to write a customized photorealistic prompt.
   - Passes your standby photo to **Qwen-Image-3.0** via Alibaba DashScope Multimodal Generation API.
   - Generates an image that keeps your facial features & identity consistent while placing you in a scene matching the post (e.g. coding workstation, university mentoring lab, Figma UI/UX review).
3. The generated image is saved to `data/generated_images/` and attached as a draft asset to Buffer!

### 💡 Tips for the Standby Photo:
- Clear, well-lit portrait or headshot/waist-up photo of yourself.
- Neutral or pleasant expression.
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`.
