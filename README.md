**This pipe integrates OpenRouter’s image-capable models (like `google/gemini-2.5-flash-image`) into OpenWebUI.**

⚠️ **Important**: When using this pipe, set an *External Task Model* in OpenWebUI settings.  
This pipe triggers on system tasks (tags, title, etc.). If those tasks also call this pipe, it can cause errors at the end — even if image generation/editing works fine.

get function: https://openwebui.com/f/panuud/openrouterimage
