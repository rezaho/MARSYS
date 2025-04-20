import asyncio
import os
import tempfile

from PIL import Image  # Ensure Pillow is installed (pip install Pillow)


class AsyncOSController:
    """
    An asynchronous OS controller using xdotool and ImageMagick's import command.
    """

    async def run_command(self, *args):
        """
        Helper method to run a subprocess command asynchronously.
        Raises an Exception if the command fails.
        """
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise Exception(
                f"Command {' '.join(args)} failed: {stderr.decode().strip()}"
            )
        return stdout.decode().strip()

    async def key(self, key_str):
        """Simulate a key press."""
        await self.run_command("xdotool", "key", key_str)

    async def type(self, text):
        """Simulate typing a string of text."""
        await self.run_command("xdotool", "type", text)

    async def mouse_move(self, x, y):
        """Move the mouse cursor to (x, y)."""
        await self.run_command("xdotool", "mousemove", str(x), str(y))

    async def left_click(self):
        """Simulate a left mouse click."""
        await self.run_command("xdotool", "click", "1")

    async def left_click_drag(self, start_x, start_y, end_x, end_y, duration=1.0):
        """
        Simulate a left mouse click and drag from (start_x, start_y) to (end_x, end_y).
        The drag lasts for the given duration (in seconds).
        """
        # Move to the starting position
        await self.mouse_move(start_x, start_y)
        # Press and hold the left mouse button
        await self.run_command("xdotool", "mousedown", "1")
        # Gradually move the mouse to the end position in small steps
        steps = 20
        for i in range(1, steps + 1):
            x = int(start_x + (end_x - start_x) * i / steps)
            y = int(start_y + (end_y - start_y) * i / steps)
            await self.mouse_move(x, y)
            await asyncio.sleep(duration / steps)
        # Release the left mouse button
        await self.run_command("xdotool", "mouseup", "1")

    async def right_click(self):
        """Simulate a right mouse click."""
        await self.run_command("xdotool", "click", "3")

    async def middle_click(self):
        """Simulate a middle mouse click."""
        await self.run_command("xdotool", "click", "2")

    async def double_click(self):
        """Simulate a double left mouse click."""
        await self.run_command(
            "xdotool", "click", "--repeat", "2", "--delay", "100", "1"
        )

    async def triple_click(self):
        """Simulate a triple left mouse click."""
        await self.run_command(
            "xdotool", "click", "--repeat", "3", "--delay", "100", "1"
        )

    async def screenshot(self, output_file=None):
        """
        Capture a screenshot of the current display using ImageMagick's import command.
        If output_file is provided, the screenshot is saved there.
        Otherwise, a temporary file is used and the method returns a PIL.Image object.
        """
        if output_file is None:
            # Create a temporary file for the screenshot
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            await self.run_command("import", "-window", "root", tmp_path)
            if os.path.exists(tmp_path):
                img = Image.open(tmp_path)
                os.remove(tmp_path)
                return img
            else:
                return None
        else:
            await self.run_command("import", "-window", "root", output_file)
            if os.path.exists(output_file):
                img = Image.open(output_file)
                return img
            else:
                return None

    async def cursor_position(self):
        """
        Get the current mouse cursor position.
        Returns a tuple: (x, y).
        """
        output = await self.run_command("xdotool", "getmouselocation")
        # Expected output format: "x:100 y:200 screen:0 window:12345"
        parts = output.split()
        pos = {}
        for part in parts:
            if ":" in part:
                key, val = part.split(":")
                pos[key] = int(val)
        return (pos.get("x"), pos.get("y"))

    async def left_mouse_down(self):
        """Simulate pressing (holding down) the left mouse button."""
        await self.run_command("xdotool", "mousedown", "1")

    async def left_mouse_up(self):
        """Simulate releasing the left mouse button."""
        await self.run_command("xdotool", "mouseup", "1")

    async def scroll(self, direction, times=1):
        """
        Simulate scrolling in a given direction.
        Valid directions: "up", "down", "left", "right".
        The 'times' parameter specifies how many scroll events to send.
        """
        mapping = {"up": "4", "down": "5", "left": "6", "right": "7"}
        button = mapping.get(direction.lower())
        if button is None:
            raise ValueError(
                "Invalid scroll direction. Use 'up', 'down', 'left', or 'right'."
            )
        for _ in range(times):
            await self.run_command("xdotool", "click", button)
            await asyncio.sleep(0.1)

    async def hold_key(self, key, duration=1.0):
        """
        Hold down a key for the specified duration (in seconds).
        """
        await self.run_command("xdotool", "keydown", key)
        await asyncio.sleep(duration)
        await self.run_command("xdotool", "keyup", key)

    async def wait(self, seconds):
        """Wait for the specified number of seconds."""
        await asyncio.sleep(seconds)

        await asyncio.sleep(seconds)
