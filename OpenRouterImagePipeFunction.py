import random    
import base64    
import asyncio    
import re    
import requests    
from typing import List, AsyncGenerator, Callable, Awaitable    
from pydantic import BaseModel, Field    
    
    
class Pipe:    
    class Valves(BaseModel):    
        OPENROUTER_API_KEYS: str = Field(    
            default="", description="OpenRouter API Keys, comma-separated"    
        )    
        IMAGE_NUM: int = Field(default=1, description="Number of images (1-10)")    
        IMAGE_SIZE: str = Field(    
            default="1024x1024",    
            description="Image size: 1024x1024, 1536x1024, 1024x1536, auto",    
        )    
        BASE_URL: str = Field(    
            default="https://openrouter.ai/api/v1",    
            description="OpenRouter-compatible endpoint base URL. Defaults to OpenRouter's official endpoint.",    
        )    
        IMAGE_MODEL: str = Field(default="google/gemini-2.5-flash-image", description="OpenRouter image model")    
    
    def __init__(self):    
        self.type = "manifold"    
        self.name = ""    
        self.valves = self.Valves()    
        self.emitter: Callable[[dict], Awaitable[None]] | None = None    
    
    def _get_base_url(self) -> str:    
        # Return base_url if set and non-empty, else default    
        val = getattr(self.valves, "BASE_URL", None)    
        if val is not None and len(val.strip()) > 0:    
            return val.strip()    
        return "https://openrouter.ai/api/v1"  # Fallback default for OpenRouter    
    
    async def emit_status(self, message: str = "", done: bool = False):    
        if self.emitter:    
            await self.emitter(    
                {"type": "status", "data": {"description": message, "done": done}}    
            )    
    
    def _get_aspect_ratio(self) -> str | None:    
        """Map IMAGE_SIZE to OpenRouter's image_config aspect_ratio."""    
        size_map = {    
            "1024x1024": "1:1",    
            "1536x1024": "3:2",    
            "1024x1536": "2:3",    
            "832x1216": "2:3",
            "1216x832": "3:2",
            "auto": None,    
        }    
        size = self.valves.IMAGE_SIZE.lower()    
        return size_map.get(size, "1:1")  # Default to square    
    
    async def pipes(self) -> List[dict]:    
        return [{"id": "openrouter-image", "name": "openrouter-image"}]    
    
    def convert_message_to_prompt(self, messages: List[dict]) -> tuple[str, List[dict]]:    
        for msg in reversed(messages):    
            if msg.get("role") != "user":    
                continue    
            content = msg.get("content")    
    
            # If content is a list (it can be mixed text and images)    
            if isinstance(content, list):    
                text_parts, image_data_list = [], []    
                for part in content:    
                    if part.get("type") == "text":    
                        text_parts.append(part.get("text", ""))    
                    elif part.get("type") == "image_url":    
                        url = part.get("image_url", {}).get("url", "")    
                        if url.startswith("data:"):    
                            header, data = url.split(";base64,", 1)    
                            mime = header.split("data:")[-1]    
                            image_data_list.append({"mimeType": mime, "data": data})    
                prompt = (    
                    " ".join(text_parts).strip() or "Please edit the provided image(s)"    
                )    
                return prompt, image_data_list    
    
            # If content is a plain string (search for embedded images in it)    
            if isinstance(content, str):    
                pattern = r"!\[[^\]]*\]\(data:([^;]+);base64,([^)]+)\)"    
                matches = re.findall(pattern, content)    
                image_data_list = [{"mimeType": m, "data": d} for m, d in matches]    
                clean = (    
                    re.sub(pattern, "", content).strip()    
                    or "Please edit the provided image(s)"    
                )    
                return clean, image_data_list    
    
        # Default case: No images found, return a default prompt    
        return "Please edit the provided image(s)", []    
    
    async def _run_blocking(self, fn: Callable, *args, **kwargs):    
        loop = asyncio.get_running_loop()    
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))    
    
    async def generate_image(    
        self,    
        prompt: str,    
        model: str,    
        n: int,    
    ) -> AsyncGenerator[str, None]:    
        await self.emit_status("🎨 Generating images...")    
        key = random.choice(self.valves.OPENROUTER_API_KEYS.split(",")).strip()    
        if not key:    
            yield "Error: OPENROUTER_API_KEYS not set"    
            return    
    
        base_url = self._get_base_url()    
    
        url = f"{base_url.rstrip('/')}/chat/completions"            
    
        payload = {    
            "model": model,    
            "messages": [    
                {    
                    "role": "user",    
                    "content": prompt,    
                }    
            ],    
            "modalities": ["image", "text"],    
        }    
    
        aspect = self._get_aspect_ratio()    
        if aspect:    
            payload["image_config"] = {"aspect_ratio": aspect}    
    
        headers = {    
            "Authorization": f"Bearer {key}",    
            "Content-Type": "application/json",    
        }    
    
        def _make_request():    
            response = requests.post(url, headers=headers, json=payload)    
            response.raise_for_status()  # Raises HTTPError   
            return response.json()    
    
        try:    
            resp_data = await self._run_blocking(_make_request)    
            if "choices" not in resp_data or not resp_data["choices"]:    
                yield "Error: Invalid response structure (no choices)"    
                await self.emit_status("❌ Invalid API response", done=True)    
                return    
    
            message = resp_data["choices"][0]["message"]    
    
            # Yield the assistant's content (text description) if present and non-empty    
            content = message.get("content", "").strip()    
            if content:    
                yield content    
    
            if "images" not in message or not message["images"]:    
                yield "Error: No images generated (check model/prompt/modalities)"    
                await self.emit_status("❌ No images in response", done=True)    
                return    
    
            for i, img_obj in enumerate(message["images"][:n], 1):  # Limit to n    
                url_data = img_obj.get("image_url", {}).get("url", "")    
                if not url_data.startswith("data:image/"):    
                    yield f"Warning: Invalid image URL for image {i}"    
                    continue    
                yield f"![image_{i}]({url_data})"    
    
            num_generated = min(len(message["images"]), n)    
            await self.emit_status(f"🎉 Generated {num_generated} image(s)", done=True)    
        except requests.exceptions.RequestException as e:    
            error_msg = f"HTTP Error: {e}"    
            yield error_msg    
            await self.emit_status(f"❌ {error_msg}", done=True)    
        except Exception as e:    
            yield f"Error during image generation: {e}"    
            await self.emit_status("❌ Image generation failed", done=True)    
    
    async def edit_image(    
        self,    
        base64_images: List[dict],    
        prompt: str,    
        model: str,    
        n: int,    
    ) -> AsyncGenerator[str, None]:    
        await self.emit_status("✂️ Editing images...")    
        key = random.choice(self.valves.OPENROUTER_API_KEYS.split(",")).strip()    
        if not key:    
            yield "Error: OPENROUTER_API_KEYS not set"    
            return    
    
        base_url = self._get_base_url()    
    
        url = f"{base_url.rstrip('/')}/chat/completions"    
    
        # Validate and prepare input images (build data URLs)    
        content_parts = []    
        for i, img_dict in enumerate(base64_images, start=1):    
            try:    
                mime = img_dict["mimeType"]    
                if mime not in {"image/png", "image/jpeg", "image/webp"}:    
                    raise ValueError(f"Unsupported format for image {i}: {mime}")    
    
                data_url = f"data:{mime};base64,{img_dict['data']}"    
                content_parts.append(    
                    {"type": "image_url", "image_url": {"url": data_url}}    
                )    
            except Exception as e:    
                yield f"Error processing input image {i}: {e}"    
                return  # Stop on error    
    
        if not content_parts:    
            yield "Error: No valid input images"    
            return    
    
        full_content = [{"type": "text", "text": prompt}] if prompt.strip() else []    
        full_content.extend(content_parts)  # Append image parts    
    
        payload = {    
            "model": model,    
            "messages": [    
                {    
                    "role": "user",    
                    "content": full_content,
                }    
            ],    
            "modalities": ["image", "text"],    
        }    
    
        aspect = self._get_aspect_ratio()    
        if aspect:    
            payload["image_config"] = {"aspect_ratio": aspect}    
    
        headers = {    
            "Authorization": f"Bearer {key}",    
            "Content-Type": "application/json",    
        }    
    
        def _make_request():    
            response = requests.post(url, headers=headers, json=payload)    
            response.raise_for_status()    
            return response.json()    
    
        try:    
            resp_data = await self._run_blocking(_make_request)    
            if "choices" not in resp_data or not resp_data["choices"]:    
                yield "Error: Invalid response structure (no choices)"    
                await self.emit_status("❌ Invalid API response", done=True)    
                return    
    
            message = resp_data["choices"][0]["message"]    
    
            # Yield the assistant's content (text description) if present and non-empty    
            content = message.get("content", "").strip()    
            if content:    
                yield content    
    
            if "images" not in message or not message["images"]:    
                yield "Error: No edited images generated (check model support for image input/output)"    
                await self.emit_status("❌ No images in response", done=True)    
                return    
    
            for i, img_obj in enumerate(message["images"][:n], 1):    
                url_data = img_obj.get("image_url", {}).get("url", "")    
                if not url_data.startswith("data:image/"):    
                    yield f"Warning: Invalid edited image URL for {i}"    
                    continue    
                yield f"![edited_image_{i}]({url_data})"    
    
            num_generated = min(len(message["images"]), n)    
            await self.emit_status(f"🎉 Edited {num_generated} image(s)", done=True)    
        except requests.exceptions.RequestException as e:    
            error_msg = f"HTTP Error: {e}"    
            yield error_msg    
            await self.emit_status(f"❌ {error_msg}", done=True)    
        except Exception as e:    
            yield f"Error during image edit: {e}"    
            await self.emit_status("❌ Image edit failed", done=True)    
    
    async def pipe(
        self,
        body: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> AsyncGenerator[str, None]:
        self.emitter = __event_emitter__
        msgs = body.get("messages", [])
        prompt, imgs = self.convert_message_to_prompt(msgs)

        # Check if this is a streaming continuation (no new content)
        # Actually this one does not work as I intended haha
        if not prompt and not imgs:
            return  # Skip processing for empty streaming chunks

        if imgs:
            async for out in self.edit_image(
                base64_images=imgs,
                prompt=prompt,
                model=self.valves.IMAGE_MODEL,
                n=min(max(1, self.valves.IMAGE_NUM), 4),
            ):
                yield out
        else:
            async for out in self.generate_image(
                prompt=prompt,
                model=self.valves.IMAGE_MODEL,
                n=min(max(1, self.valves.IMAGE_NUM), 4),
            ):
                yield out