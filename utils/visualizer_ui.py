import os
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk

# Configuration
BG_COLOR = "#FFFFFF"       # White background
TEXT_COLOR = "#333333"     # Dark gray text
ACCENT_COLOR = "#4CAF50"   # Green accent
FONT_MAIN = ("Helvetica", 12)
FONT_BOLD = ("Helvetica", 12, "bold")
FONT_TITLE = ("Helvetica", 16, "bold")
FONT_MONO = ("Consolas", 11)

def show_visualization_window(
    title: str,
    images: list,  # List of dicts: {"path": str, "label": str} or {"image": PIL.Image, "label": str}
    prompts: str,
    response: str,
    mode: str = "auto",
    timeout: int = 5000
):
    """
    Pop up a styled visualization window with vertically stacked images.
    """
    import time # Import locally
    try:
        # Check display
        if not os.environ.get('DISPLAY') and sys.platform != 'darwin' and sys.platform != 'win32':
            print("[Viz] Skipping: No DISPLAY env var (headless mode?)", flush=True)
            return

        root = tk.Tk()
        root.title(f"OASIS Visualizer ({mode.upper()}) - {title}")
        
        # Fullscreen mode
        root.attributes('-fullscreen', True)
        root.configure(bg=BG_COLOR)
        
        # Update to get actual dimensions
        root.update_idletasks()
        win_w = root.winfo_width()
        win_h = root.winfo_height()
        
        # Fallback if dimensions are weird
        if win_w < 100: 
            win_w = root.winfo_screenwidth()
            win_h = root.winfo_screenheight()

        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background=BG_COLOR)
        style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR, font=FONT_MAIN)
        style.configure("Header.TLabel", font=FONT_TITLE, foreground=TEXT_COLOR)

        # --- Layout ---
        # 0. Header
        header_frame = tk.Frame(root, bg=BG_COLOR, pady=10, padx=20)
        header_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Clean Title (No debug suffixes)
        tk.Label(header_frame, text=title, font=FONT_TITLE, bg=BG_COLOR, fg=TEXT_COLOR).pack(side=tk.LEFT)
        
        timer_lbl = tk.Label(header_frame, text="", font=FONT_BOLD, bg=BG_COLOR, fg="#E53935")
        timer_lbl.pack(side=tk.RIGHT)
        
        if mode == "auto":
             timer_lbl.config(text=f"Auto-closing in {timeout/1000}s")
        # Manual mode: No text label needed, the button at bottom is enough

        # Main Container (Left: Images, Right: Text)
        content_frame = tk.Frame(root, bg=BG_COLOR)
        content_frame.pack(fill=tk.BOTH, expand=True)

        main_paned = tk.PanedWindow(content_frame, orient=tk.HORIZONTAL, bg=BG_COLOR, sashwidth=6, sashrelief=tk.FLAT)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Left: Images Column (Scrollable) ---
        # Give it 50% weight/width
        image_container = tk.Frame(main_paned, bg=BG_COLOR)
        main_paned.add(image_container, minsize=300, stretch="always", width=int(win_w * 0.5))

        # Canvas for scrolling
        canvas = tk.Canvas(image_container, bg=BG_COLOR, highlightthickness=0)
        scrollbar = ttk.Scrollbar(image_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=BG_COLOR)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        # Mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel) # Windows
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units")) # Linux
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))  # Linux

        window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # Ensure scrollable frame fits canvas width
        def _configure_window(event):
            canvas.itemconfig(window_id, width=event.width)
        canvas.bind("<Configure>", _configure_window)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Force update to get canvas width
        root.update_idletasks()
        
        # Target width: 50% of actual window width minus margin
        target_content_width = int(win_w * 0.5) - 60

        # Load and display images vertically
        loaded_photos = [] 
        
        if not images:
             tk.Label(scrollable_frame, text="(No Images Provided)", bg=BG_COLOR, font=FONT_MAIN, fg="#888").pack(pady=20)
        
        for idx, item in enumerate(images):
            try:
                img_obj = None
                label_text = item.get("label", f"Image {idx+1}")
                
                if "path" in item and item["path"]:
                    if os.path.exists(item["path"]):
                        img_obj = Image.open(item["path"])
                    else:
                        label_text += " (File Not Found)"
                elif "image" in item and item["image"]:
                    img_obj = item["image"]
                
                # Container for each image/text block
                block_frame = tk.Frame(scrollable_frame, bg=BG_COLOR, pady=10, padx=5)
                block_frame.pack(fill=tk.X)
                
                # Label above item
                tk.Label(block_frame, text=f"#{idx+1} {label_text}", font=FONT_BOLD, bg=BG_COLOR, anchor="w").pack(fill=tk.X, pady=(0, 5))
                
                # Check for Text Content (e.g. Anchor Info)
                if "text_content" in item and item["text_content"]:
                    # FIX 3: Use tk.Text instead of Label, same as right side logic
                    # Calculate approximate height based on line count
                    content = item["text_content"]
                    line_count = content.count('\n') + 2
                    
                    txt_widget = tk.Text(block_frame, height=line_count, width=80, # Width is chars, controlled by pack fill
                                         font=FONT_MONO, bg=BG_COLOR, fg=TEXT_COLOR, 
                                         bd=0, highlightthickness=0, relief=tk.FLAT, wrap=tk.WORD)
                    txt_widget.pack(fill=tk.X, anchor="w", padx=5)
                    txt_widget.insert(tk.END, content)
                    txt_widget.configure(state='disabled') # Read-only
                
                elif img_obj:
                    # Resize logic: Fill the target width
                    orig_w, orig_h = img_obj.size
                    
                    target_w = target_content_width
                    
                    ratio = target_w / float(orig_w)
                    target_h = int(float(orig_h) * ratio)
                    
                    img_resized = img_obj.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img_resized)
                    loaded_photos.append(photo)
                    
                    img_lbl = tk.Label(block_frame, image=photo, bg="#F0F0F0", bd=1, relief=tk.SOLID)
                    img_lbl.pack(anchor="center")
                else:
                    tk.Label(block_frame, text="[Image Data Missing]", bg="#FFEBEE", fg="red").pack(pady=10)
                    
                # Separator
                ttk.Separator(scrollable_frame, orient='horizontal').pack(fill='x', padx=20, pady=5)
                
            except Exception as e:
                print(f"[Viz] Error loading image {idx}: {e}")
                tk.Label(scrollable_frame, text=f"Error loading image: {e}", fg="red", bg=BG_COLOR).pack()

        # --- Right: Text Column (Prompts & Response) ---
        text_container = tk.Frame(main_paned, bg=BG_COLOR, padx=10)
        # Give it 50% weight/width too, and allow stretch
        main_paned.add(text_container, minsize=300, stretch="always", width=int(win_w * 0.5))

        # Prompts Area
        lbl_prompts = tk.Label(text_container, text="Task / Context", font=FONT_BOLD, bg=BG_COLOR, fg="#1976D2", anchor="w")
        lbl_prompts.pack(fill=tk.X, pady=(0, 5))
        
        # Use larger font for simplified text
        txt_prompt = tk.Text(text_container, height=6, font=("Helvetica", 14), bg="#FAFAFA", relief=tk.FLAT, bd=1, wrap=tk.WORD)
        txt_prompt.pack(fill=tk.X, pady=(0, 20))
        txt_prompt.insert(tk.END, prompts)
        # txt_prompt.configure(state='disabled') # Read-only

        # Response Area
        lbl_resp = tk.Label(text_container, text="VLM Decision", font=FONT_BOLD, bg=BG_COLOR, fg="#388E3C", anchor="w")
        lbl_resp.pack(fill=tk.X, pady=(0, 5))
        
        txt_resp = tk.Text(text_container, height=15, font=("Helvetica", 14), bg="#F1F8E9", relief=tk.FLAT, bd=1, wrap=tk.WORD)
        txt_resp.pack(fill=tk.BOTH, expand=True)
        txt_resp.insert(tk.END, response)

        # --- Mode Logic ---
        if mode == "manual":
            # Add Continue Button at Bottom
            btn_frame = tk.Frame(root, bg=BG_COLOR, pady=15)
            btn_frame.pack(side=tk.BOTTOM, fill=tk.X)
            
            btn = tk.Button(btn_frame, text="CONTINUE ➔", command=root.destroy, 
                           bg=ACCENT_COLOR, fg="white", font=("Helvetica", 14, "bold"), 
                           relief=tk.FLAT, padx=30, pady=10)
            btn.pack()
            
            # Key binding for Enter/Space to continue
            root.bind('<Return>', lambda e: root.destroy())
            root.bind('<space>', lambda e: root.destroy())
            
        else: # Auto mode
            start_time = time.time()
            duration_sec = timeout / 1000.0
            
            def update_timer():
                elapsed = time.time() - start_time
                remaining = duration_sec - elapsed
                
                if remaining <= 0:
                    root.destroy()
                else:
                    timer_lbl.config(text=f"Auto-closing in {remaining:.1f}s")
                    root.after(100, update_timer)
            
            root.after(100, update_timer)
        
        # Center and Focus
        root.update_idletasks()
        root.lift()
        root.attributes('-topmost', True)
        root.after_idle(root.attributes, '-topmost', False)
        
        print(f"[Viz] Window '{title}' opened (Mode: {mode}).", flush=True)
        root.mainloop()
        print(f"[Viz] Window closed.", flush=True)

    except Exception as e:
        print(f"[Viz] Visualization failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
