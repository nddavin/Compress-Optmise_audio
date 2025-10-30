import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import subprocess
import os
import sys
import logging
import time
from pathlib import Path
from typing import Any

class AudioCompressorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Pure Sound - Professional Audio Processing")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        # Import here to avoid circular imports and improve startup time
        from compress_audio import (
            get_format_defaults, get_compressor_preset, get_multiband_preset,
            build_audio_filters, compress_audio, create_output_dirs,
            process_files, check_ffmpeg, setup_logging,
            _get_config_manager
        )
        from presets import preset_manager
        # Lazy load config_manager
        config_manager = _get_config_manager()

        # Make preset_manager available globally in this class
        self.preset_manager = preset_manager

        self.compress_audio = compress_audio
        self.process_files = process_files
        self.create_output_dirs = create_output_dirs
        self.build_audio_filters = build_audio_filters
        self.config_manager = config_manager

        # Initialize variables
        self.input_dir = tk.StringVar(value=".")
        self.output_dir = tk.StringVar(value="./optimized")
        self.format_var = tk.StringVar(value="mp3")
        self.content_type = tk.StringVar(value="speech")
        self.bitrates = tk.StringVar(value="64,96,128")

        # Compressor settings variables
        self.comp_threshold_var = tk.DoubleVar(value=-20)
        self.comp_ratio_var = tk.DoubleVar(value=3)
        self.comp_attack_var = tk.DoubleVar(value=0.01)
        self.comp_release_var = tk.DoubleVar(value=0.1)
        self.comp_makeup_var = tk.DoubleVar(value=6)

        # Multiband settings variables
        self.mb_low_freq_var = tk.IntVar(value=250)
        self.mb_high_freq_var = tk.IntVar(value=4000)

        # Processing state
        self.is_processing = False
        self.current_process = None
        self.current_recommendations = None

        self.setup_ui()
        self.load_config()

    def setup_ui(self):
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Main processing tab
        main_frame = ttk.Frame(notebook)
        notebook.add(main_frame, text="Compress Audio")

        # Settings tab
        settings_frame = ttk.Frame(notebook)
        notebook.add(settings_frame, text="Settings")

        # Preview tab
        preview_frame = ttk.Frame(notebook)
        notebook.add(preview_frame, text="Preview")

        self.setup_main_tab(main_frame)
        self.setup_settings_tab(settings_frame)
        self.setup_preview_tab(preview_frame)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill='x', padx=10, pady=(0, 10))

    def setup_main_tab(self, parent: ttk.Frame):
        # Input/Output section
        io_frame = ttk.LabelFrame(parent, text="Input/Output", padding=10)
        io_frame.pack(fill='x', padx=10, pady=5)

        # Input directory
        ttk.Label(io_frame, text="Input Directory:").grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(io_frame, textvariable=self.input_dir, width=50).grid(row=0, column=1, padx=(10,5), pady=2)
        ttk.Button(io_frame, text="Browse", command=self.browse_input).grid(row=0, column=2, pady=2)

        # Output directory
        ttk.Label(io_frame, text="Output Directory:").grid(row=1, column=0, sticky='w', pady=2)
        ttk.Entry(io_frame, textvariable=self.output_dir, width=50).grid(row=1, column=1, padx=(10,5), pady=2)
        ttk.Button(io_frame, text="Browse", command=self.browse_output).grid(row=1, column=2, pady=2)

        # Format and content type
        format_frame = ttk.Frame(parent)
        format_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(format_frame, text="Output Format:").grid(row=0, column=0, sticky='w', pady=2)
        format_combo = ttk.Combobox(format_frame, textvariable=self.format_var,
                                   values=["mp3", "aac", "ogg", "opus", "flac"], state="readonly")
        format_combo.grid(row=0, column=1, padx=(10,20), pady=2)
        format_combo.bind('<<ComboboxSelected>>', self.on_format_change)

        ttk.Label(format_frame, text="Content Type:").grid(row=0, column=2, sticky='w', pady=2)
        content_combo = ttk.Combobox(format_frame, textvariable=self.content_type,
                                    values=["speech", "music"], state="readonly")
        content_combo.grid(row=0, column=3, padx=(10,20), pady=2)
        content_combo.bind('<<ComboboxSelected>>', self.on_content_type_change)

        ttk.Label(format_frame, text="Bitrates (comma-separated):").grid(row=0, column=4, sticky='w', pady=2)
        ttk.Entry(format_frame, textvariable=self.bitrates, width=20).grid(row=0, column=5, padx=(10,0), pady=2)

        # Audio processing options
        processing_frame = ttk.LabelFrame(parent, text="Audio Processing", padding=10)
        processing_frame.pack(fill='x', padx=10, pady=5)

        # Checkboxes for processing options
        self.normalize_var = tk.BooleanVar(value=True)
        self.compressor_var = tk.BooleanVar(value=False)
        self.multiband_var = tk.BooleanVar(value=False)
        self.ml_noise_var = tk.BooleanVar(value=False)
        self.silence_trim_var = tk.BooleanVar(value=False)
        self.noise_gate_var = tk.BooleanVar(value=False)
        self.parallel_var = tk.BooleanVar(value=False)
        self.preview_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(processing_frame, text="Loudness Normalization", variable=self.normalize_var).grid(row=0, column=0, sticky='w', padx=(0,20))
        ttk.Checkbutton(processing_frame, text="Dynamic Compression", variable=self.compressor_var).grid(row=0, column=1, sticky='w', padx=(0,20))
        ttk.Checkbutton(processing_frame, text="Multiband Compression", variable=self.multiband_var).grid(row=0, column=2, sticky='w', padx=(0,20))

        ttk.Checkbutton(processing_frame, text="ML Noise Reduction", variable=self.ml_noise_var).grid(row=1, column=0, sticky='w', padx=(0,20))
        ttk.Checkbutton(processing_frame, text="Silence Trimming", variable=self.silence_trim_var).grid(row=1, column=1, sticky='w', padx=(0,20))
        ttk.Checkbutton(processing_frame, text="Noise Gating", variable=self.noise_gate_var).grid(row=1, column=2, sticky='w', padx=(0,20))

        ttk.Checkbutton(processing_frame, text="Parallel Processing", variable=self.parallel_var).grid(row=2, column=0, sticky='w', padx=(0,20))
        ttk.Checkbutton(processing_frame, text="Generate Preview", variable=self.preview_var).grid(row=2, column=1, sticky='w', padx=(0,20))

        # Preset buttons
        preset_frame = ttk.LabelFrame(parent, text="Quick Presets", padding=10)
        preset_frame.pack(fill='x', padx=10, pady=5)

        # Create preset buttons in a grid
        presets = self.preset_manager.get_all_presets()
        for i, preset in enumerate(presets[:6]):  # Show first 6 presets
            btn = ttk.Button(preset_frame, text=f"{preset.icon} {preset.name}",
                           command=lambda p=preset: self.apply_preset(p))
            btn.grid(row=i//3, column=i%3, padx=5, pady=2, sticky='ew')
            preset_frame.grid_columnconfigure(i%3, weight=1)

        # Custom workflows section
        workflow_frame = ttk.LabelFrame(parent, text="Custom Workflows", padding=10)
        workflow_frame.pack(fill='x', padx=10, pady=5)

        # Workflow buttons
        workflows = self.preset_manager.get_all_custom_workflows()
        for i, workflow in enumerate(workflows[:3]):  # Show first 3 workflows
            btn = ttk.Button(workflow_frame, text=f"{workflow.icon} {workflow.name}",
                           command=lambda w=workflow: self.execute_custom_workflow(w))
            btn.grid(row=0, column=i, padx=5, pady=2, sticky='ew')
            workflow_frame.grid_columnconfigure(i, weight=1)

        # Workflow management buttons
        ttk.Button(workflow_frame, text="Create Workflow", command=self.create_custom_workflow).grid(row=1, column=0, padx=5, pady=2, sticky='ew')
        ttk.Button(workflow_frame, text="Manage Workflows", command=self.manage_workflows).grid(row=1, column=1, padx=5, pady=2, sticky='ew')
        ttk.Button(workflow_frame, text="Smart Suggestions", command=self.show_smart_suggestions).grid(row=1, column=2, padx=5, pady=2, sticky='ew')

        # Control buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill='x', padx=10, pady=10)

        self.start_btn = ttk.Button(button_frame, text="Start Compression", command=self.start_compression)
        self.start_btn.pack(side='left', padx=(0,10))

        self.stop_btn = ttk.Button(button_frame, text="Stop", command=self.stop_compression, state='disabled')
        self.stop_btn.pack(side='left', padx=(0,10))

        ttk.Button(button_frame, text="Analyze Files", command=self.analyze_files).pack(side='left', padx=(0,10))
        ttk.Button(button_frame, text="Save Settings", command=self.save_config).pack(side='left', padx=(0,10))
        ttk.Button(button_frame, text="Queue with Preset", command=self.queue_with_preset).pack(side='left')

        # Progress and log
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding=10)
        progress_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', pady=(0,10))

        self.log_text = scrolledtext.ScrolledText(progress_frame, height=10, wrap=tk.WORD)
        self.log_text.pack(fill='both', expand=True)

    def setup_settings_tab(self, parent: ttk.Frame):
        # Compressor settings
        comp_frame = ttk.LabelFrame(parent, text="Compressor Settings", padding=10)
        comp_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(comp_frame, text="Preset:").grid(row=0, column=0, sticky='w', pady=2)
        self.comp_preset_var = tk.StringVar(value="speech")
        comp_preset_combo = ttk.Combobox(comp_frame, textvariable=self.comp_preset_var,
                                        values=["speech", "music", "broadcast", "gentle"], state="readonly")
        comp_preset_combo.grid(row=0, column=1, padx=(10,20), pady=2)

        # Threshold, Ratio, Attack, Release, Makeup sliders
        self.setup_slider(comp_frame, "Threshold (dB):", self.comp_threshold_var, -30, 0, -20, 1, 0, 2)
        self.setup_slider(comp_frame, "Ratio:", self.comp_ratio_var, 1, 20, 3, 2, 1, 3)
        self.setup_slider(comp_frame, "Attack (s):", self.comp_attack_var, 0.001, 1, 0.01, 3, 2, 0)
        self.setup_slider(comp_frame, "Release (s):", self.comp_release_var, 0.01, 2, 0.1, 4, 3, 1)
        self.setup_slider(comp_frame, "Makeup (dB):", self.comp_makeup_var, 0, 20, 6, 5, 4, 2)

        # Multiband settings
        mb_frame = ttk.LabelFrame(parent, text="Multiband Settings", padding=10)
        mb_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(mb_frame, text="Preset:").grid(row=0, column=0, sticky='w', pady=2)
        self.mb_preset_var = tk.StringVar(value="speech")
        mb_preset_combo = ttk.Combobox(mb_frame, textvariable=self.mb_preset_var,
                                      values=["speech", "music", "vocal"], state="readonly")
        mb_preset_combo.grid(row=0, column=1, padx=(10,20), pady=2)

        # Frequency crossovers
        self.setup_slider(mb_frame, "Low/Mid Crossover (Hz):", self.mb_low_freq_var, 100, 1000, 250, 1, 2, 0)
        self.setup_slider(mb_frame, "Mid/High Crossover (Hz):", self.mb_high_freq_var, 1000, 10000, 4000, 2, 3, 1)

        # Channel settings
        channel_frame = ttk.LabelFrame(parent, text="Channel Settings", padding=10)
        channel_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(channel_frame, text="Channels:").grid(row=0, column=0, sticky='w', pady=2)
        self.channels_var = tk.IntVar(value=1)
        channels_combo = ttk.Combobox(channel_frame, textvariable=self.channels_var,
                                     values=["1", "2", "6", "8"], state="readonly")
        channels_combo.grid(row=0, column=1, padx=(10,20), pady=2)

        ttk.Label(channel_frame, text="Channel Layout:").grid(row=0, column=2, sticky='w', pady=2)
        self.channel_layout_var = tk.StringVar(value="mono")
        layout_combo = ttk.Combobox(channel_frame, textvariable=self.channel_layout_var,
                                   values=["mono", "stereo", "5.1", "7.1"], state="readonly")
        layout_combo.grid(row=0, column=3, padx=(10,20), pady=2)

        self.downmix_var = tk.BooleanVar(value=False)
        self.upmix_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(channel_frame, text="Downmix to Stereo", variable=self.downmix_var).grid(row=1, column=0, sticky='w', padx=(0,20))
        ttk.Checkbutton(channel_frame, text="Upmix to Surround", variable=self.upmix_var).grid(row=1, column=1, sticky='w', padx=(0,20))

    def setup_slider(self, parent: Any, label: str, var: Any, min_val: float, max_val: float, default: float, row: int, col: int, col_span: int = 1):
        ttk.Label(parent, text=label).grid(row=row, column=col*2, sticky='w', pady=2)
        var.set(default)
        scale = tk.Scale(parent, from_=min_val, to=max_val, resolution=0.1, orient=tk.HORIZONTAL, variable=var)
        scale.grid(row=row, column=col*2+1, sticky='ew', padx=(10,0), pady=2)
        if col_span > 1:
            parent.grid_columnconfigure(col*2+1, weight=1)

    def setup_preview_tab(self, parent: ttk.Frame):
        # Preview controls
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill='x', padx=10, pady=10)

        ttk.Button(control_frame, text="Generate Preview", command=self.generate_preview).pack(side='left', padx=(0,10))
        ttk.Button(control_frame, text="Play Preview", command=self.play_preview).pack(side='left', padx=(0,10))

        # Preview visualization area
        preview_frame = ttk.LabelFrame(parent, text="Audio Analysis & Recommendations", padding=10)
        preview_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.preview_text = scrolledtext.ScrolledText(preview_frame, height=20, wrap=tk.WORD)
        self.preview_text.pack(fill='both', expand=True)

        # Add apply recommendations button
        apply_frame = ttk.Frame(parent)
        apply_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(apply_frame, text="Apply Recommendations to Settings",
                  command=self.apply_recommendations_to_settings).pack(side='left', padx=(0,10))
        ttk.Button(apply_frame, text="Clear Analysis",
                  command=lambda: self.preview_text.delete(1.0, tk.END)).pack(side='left', padx=(0,10))
        ttk.Button(apply_frame, text="Generate Preview Clip",
                  command=self.generate_preview_clip).pack(side='left')

    def browse_input(self):
        directory = filedialog.askdirectory(title="Select Input Directory")
        if directory:
            self.input_dir.set(directory)

    def browse_output(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)

    def on_format_change(self, event: Any = None):
        self.update_bitrates()

    def on_content_type_change(self, event: Any = None):
        self.update_bitrates()

    def update_bitrates(self):
        from compress_audio import get_format_defaults
        format_defaults = get_format_defaults(self.format_var.get(), self.content_type.get())
        default_bitrates = format_defaults.get(self.content_type.get(), [128])
        self.bitrates.set(','.join(map(str, default_bitrates)))

    def start_compression(self):
        if self.is_processing:
            return

        # Validate inputs
        if not os.path.exists(self.input_dir.get()):
            messagebox.showerror("Error", "Input directory does not exist!")
            return

        # Start processing in background thread
        self.is_processing = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.progress_var.set(0)
        self.log_text.delete(1.0, tk.END)

        threading.Thread(target=self.run_compression, daemon=True).start()

    def stop_compression(self):
        self.is_processing = False
        if self.current_process:
            self.current_process.terminate()
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_var.set("Stopped")

    def run_compression(self):
        try:
            self.status_var.set("Processing...")

            # Validate inputs
            input_dir = self.input_dir.get()
            if not os.path.exists(input_dir):
                raise ValueError(f"Input directory does not exist: {input_dir}")

            output_dir = self.output_dir.get()
            if not output_dir.strip():
                raise ValueError("Output directory cannot be empty")

            # Build filter chain with error handling
            try:
                filter_chain = self.build_audio_filters(
                    loudnorm_enabled=self.normalize_var.get(),
                    compressor_enabled=self.compressor_var.get(),
                    compressor_preset=self.comp_preset_var.get(),
                    multiband_enabled=self.multiband_var.get(),
                    multiband_preset=self.mb_preset_var.get(),
                    ml_noise_reduction=self.ml_noise_var.get(),
                    silence_trim_enabled=self.silence_trim_var.get(),
                    noise_gate_enabled=self.noise_gate_var.get(),
                    channels=self.channels_var.get(),
                    channel_layout=self.channel_layout_var.get() if self.channels_var.get() > 2 else None,
                    downmix=self.downmix_var.get(),
                    upmix=self.upmix_var.get()
                )
            except Exception as e:
                logging.warning(f"Failed to build audio filters: {e}. Continuing without filters.")
                filter_chain = None

            # Get bitrates with validation
            try:
                bitrates_str = self.bitrates.get().strip()
                if not bitrates_str:
                    raise ValueError("Bitrates cannot be empty")
                bitrates = [int(b.strip()) for b in bitrates_str.split(',') if b.strip()]
                if not bitrates:
                    raise ValueError("No valid bitrates specified")
            except ValueError as e:
                raise ValueError(f"Invalid bitrates: {e}")

            # Create output directories with error handling
            try:
                output_dirs = self.create_output_dirs(output_dir, bitrates)
            except Exception as e:
                raise RuntimeError(f"Failed to create output directories: {e}")

            # Process files
            extensions = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus")

            start_time = time.time()
            try:
                processed, failed, total_input_size, total_output_size = self.process_files(
                    input_dir, output_dirs, extensions, bitrates, filter_chain,
                    self.format_var.get(), self.channels_var.get(), True, False, self.parallel_var.get()
                )
            except Exception as e:
                raise RuntimeError(f"File processing failed: {e}")

            # Update UI with progress
            self.root.after(0, lambda: self.update_results(processed, failed, total_input_size, total_output_size, start_time))

        except ValueError as e:
            self.root.after(0, lambda err=e: messagebox.showerror("Input Error", str(err)))
        except RuntimeError as e:
            self.root.after(0, lambda err=e: messagebox.showerror("Processing Error", str(err)))
        except Exception as e:
            self.root.after(0, lambda err=e: messagebox.showerror("Unexpected Error", f"An unexpected error occurred: {str(err)}"))
        finally:
            self.root.after(0, self.reset_ui)

    def update_results(self, processed: int, failed: int, total_input_size: Any, total_output_size: Any, start_time: float):
        total_time = time.time() - start_time

        result_text = f"✅ Processing complete!\n"
        result_text += f"Files processed: {processed}\n"
        if failed > 0:
            result_text += f"Files failed: {failed}\n"
        result_text += f"Total time: {total_time:.2f} seconds\n"

        if processed > 0 and total_input_size > 0:
            compression_ratio = (1 - total_output_size / total_input_size) * 100
            avg_time_per_file = total_time / processed
            result_text += f"Average compression ratio: {compression_ratio:.1f}%\n"
            result_text += f"Average time per file: {avg_time_per_file:.2f} seconds\n"
            result_text += f"Total size reduction: {(total_input_size - total_output_size) / 1024 / 1024:.1f} MB\n"

        self.log_text.insert(tk.END, result_text)
        self.progress_var.set(100)

    def reset_ui(self):
        self.is_processing = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_var.set("Ready")

    def generate_preview(self):
        # Generate a short preview clip for testing settings
        self.preview_text.delete(1.0, tk.END)
        self.preview_text.insert(tk.END, "🔍 Analyzing audio files...\n\n")

        # Lazy load audio_analyzer
        from compress_audio import _get_audio_analyzer
        audio_analyzer = _get_audio_analyzer()

        try:
            # Find first audio file in input directory
            input_dir = self.input_dir.get()
            if not os.path.exists(input_dir):
                self.preview_text.insert(tk.END, "❌ Input directory does not exist!\n")
                return

            extensions = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus")
            audio_file = None

            for file in os.listdir(input_dir):
                if file.lower().endswith(extensions):
                    audio_file = os.path.join(input_dir, file)
                    break

            if not audio_file:
                self.preview_text.insert(tk.END, "❌ No audio files found in input directory!\n")
                return

            # Get quick stats
            self.preview_text.insert(tk.END, f"📊 Analyzing: {os.path.basename(audio_file)}\n\n")

            quick_stats = audio_analyzer.get_quick_stats(audio_file)
            if quick_stats:
                self.preview_text.insert(tk.END, "📈 Basic Statistics:\n")
                self.preview_text.insert(tk.END, f"   Codec: {quick_stats['codec']}\n")
                self.preview_text.insert(tk.END, f"   Sample Rate: {quick_stats['sample_rate']} Hz\n")
                self.preview_text.insert(tk.END, f"   Channels: {quick_stats['channels']}\n")
                self.preview_text.insert(tk.END, f"   Duration: {quick_stats['duration']:.1f} seconds\n")
                self.preview_text.insert(tk.END, f"   Bitrate: {quick_stats['bitrate_kbps']:.1f} kbps\n")
                self.preview_text.insert(tk.END, f"   Size: {quick_stats['size_mb']:.1f} MB\n\n")

                # Get full analysis
                analysis = audio_analyzer.analyze_file(audio_file)
                if analysis:
                    content = analysis["content_analysis"]
                    recommendations = analysis["recommendations"]

                    self.preview_text.insert(tk.END, "🎵 Content Analysis:\n")
                    self.preview_text.insert(tk.END, f"   Content Type: {content.get('content_type', 'unknown')}\n")
                    self.preview_text.insert(tk.END, f"   Dynamic Range: {content.get('dynamic_range', 'unknown')}\n")
                    self.preview_text.insert(tk.END, f"   Speech Probability: {content.get('speech_probability', 0):.2f}\n")
                    self.preview_text.insert(tk.END, f"   Music Probability: {content.get('music_probability', 0):.2f}\n\n")

                    self.preview_text.insert(tk.END, "💡 Recommendations:\n")
                    self.preview_text.insert(tk.END, f"   Format: {recommendations.get('format', 'mp3')}\n")
                    self.preview_text.insert(tk.END, f"   Bitrates: {', '.join(map(str, recommendations.get('bitrates', [128])))} kbps\n")
                    self.preview_text.insert(tk.END, f"   Enable Compression: {recommendations.get('enable_compression', False)}\n")
                    self.preview_text.insert(tk.END, f"   Enable Loudness Norm: {recommendations.get('enable_loudnorm', True)}\n\n")

                    for reason in recommendations.get('reasoning', []):
                        self.preview_text.insert(tk.END, f"   • {reason}\n")

                    # Store recommendations for later use
                    self.current_recommendations = recommendations

                    self.preview_text.insert(tk.END, "\n✅ Analysis complete! Use 'Apply Recommendations' to update settings.\n")
                else:
                    self.preview_text.insert(tk.END, "⚠️  Could not perform detailed analysis.\n")
            else:
                self.preview_text.insert(tk.END, "❌ Failed to get audio statistics.\n")

        except Exception as e:
            self.preview_text.insert(tk.END, f"❌ Analysis failed: {str(e)}\n")

    def play_preview(self):
        # Play the generated preview file
        try:
            import subprocess
            import platform

            preview_dir = os.path.join(self.output_dir.get(), "previews")
            if not os.path.exists(preview_dir):
                messagebox.showwarning("No Preview", "No preview files found. Generate a preview first.")
                return

            # Find preview file
            preview_file = None
            for file in os.listdir(preview_dir):
                if file.endswith(('.mp3', '.aac', '.ogg', '.opus', '.flac')):
                    preview_file = os.path.join(preview_dir, file)
                    break

            if not preview_file:
                messagebox.showwarning("No Preview", "No preview files found. Generate a preview first.")
                return

            # Play based on platform
            system = platform.system().lower()
            if system == "darwin":  # macOS
                subprocess.run(["afplay", preview_file], check=True)
            elif system == "linux":
                subprocess.run(["xdg-open", preview_file], check=True)
            elif system == "windows":
                import os
                os.startfile(preview_file)
            else:
                messagebox.showinfo("Preview", f"Preview file created: {preview_file}\nOpen it with your preferred audio player.")

        except subprocess.CalledProcessError:
            messagebox.showerror("Playback Error", "Audio player not found on this system.")
        except Exception as e:
            messagebox.showerror("Playback Error", f"Could not play preview: {str(e)}")

    def generate_preview_clip(self):
        """Generate a short preview clip with current settings"""
        try:
            input_dir = self.input_dir.get()
            if not os.path.exists(input_dir):
                messagebox.showerror("Error", "Input directory does not exist!")
                return

            # Find first audio file
            extensions = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus")
            audio_file = None
            for file in os.listdir(input_dir):
                if file.lower().endswith(extensions):
                    audio_file = os.path.join(input_dir, file)
                    break

            if not audio_file:
                messagebox.showerror("Error", "No audio files found in input directory!")
                return

            # Build filter chain with current settings
            filter_chain = self.build_audio_filters(
                loudnorm_enabled=self.normalize_var.get(),
                compressor_enabled=self.compressor_var.get(),
                compressor_preset=self.comp_preset_var.get(),
                multiband_enabled=self.multiband_var.get(),
                multiband_preset=self.mb_preset_var.get(),
                ml_noise_reduction=self.ml_noise_var.get(),
                silence_trim_enabled=self.silence_trim_var.get(),
                noise_gate_enabled=self.noise_gate_var.get(),
                channels=self.channels_var.get(),
                channel_layout=self.channel_layout_var.get() if self.channels_var.get() > 2 else None,
                downmix=self.downmix_var.get(),
                upmix=self.upmix_var.get()
            )

            # Create preview directory
            preview_dir = os.path.join(self.output_dir.get(), "previews")
            os.makedirs(preview_dir, exist_ok=True)

            # Generate preview clip
            filename = os.path.splitext(os.path.basename(audio_file))[0]
            preview_file = os.path.join(preview_dir, f"{filename}_preview.{self.format_var.get()}")

            success, _, _ = self.compress_audio(
                input_file=audio_file,
                output_file=preview_file,
                bitrate=128,  # Use standard bitrate for preview
                filter_chain=filter_chain,
                output_format=self.format_var.get(),
                channels=self.channels_var.get(),
                preserve_metadata=True,
                dry_run=False,
                preview_mode=True
            )

            if success:
                messagebox.showinfo("Success", f"Preview clip generated: {preview_file}")
                # Switch to preview tab to show results
                self.notebook.select(2)
                self.preview_text.insert(tk.END, f"\n🎵 Preview clip generated: {os.path.basename(preview_file)}\n")
            else:
                messagebox.showerror("Error", "Failed to generate preview clip!")

        except Exception as e:
            messagebox.showerror("Error", f"Preview generation failed: {str(e)}")

    def apply_preset(self, preset: Any):
        """Apply a workflow preset to the current settings"""
        try:
            from presets import preset_manager
            gui_settings = preset_manager.apply_preset_to_gui(preset)

            # Apply settings to GUI variables
            for attr, value in gui_settings.items():
                if hasattr(self, attr):
                    var = getattr(self, attr)
                    if hasattr(var, 'set'):
                        var.set(value)

            # Update dependent fields
            self.on_format_change()
            self.on_content_type_change()

            messagebox.showinfo("Success", f"Preset '{preset.name}' applied successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Could not apply preset: {str(e)}")

    def apply_recommendations_to_settings(self):
        """Apply the recommendations from the preview analysis to the current settings"""
        try:
            if not self.current_recommendations:
                messagebox.showwarning("No Recommendations", "Please generate a preview analysis first.")
                return

            # Apply recommendations to GUI settings
            recommendations = self.current_recommendations

            self.format_var.set(recommendations.get('format', 'mp3'))
            self.bitrates.set(','.join(map(str, recommendations.get('bitrates', [128]))))
            self.normalize_var.set(recommendations.get('enable_loudnorm', True))
            self.compressor_var.set(recommendations.get('enable_compression', False))

            messagebox.showinfo("Success", "Recommendations applied to settings!")

        except Exception as e:
            messagebox.showerror("Error", f"Could not apply recommendations: {str(e)}")

    def execute_custom_workflow(self, workflow: Any):
        """Execute a custom workflow"""
        try:
            input_dir = self.input_dir.get()
            output_dir = self.output_dir.get()

            if not os.path.exists(input_dir):
                messagebox.showerror("Error", "Input directory does not exist!")
                return

            # Find audio files
            extensions = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus")
            input_files = []
            for file in os.listdir(input_dir):
                if file.lower().endswith(extensions):
                    input_files.append(os.path.join(input_dir, file))

            if not input_files:
                messagebox.showerror("Error", "No audio files found in input directory!")
                return

            # Execute workflow
            result = self.preset_manager.execute_custom_workflow(
                workflow.id, input_files, output_dir,
                progress_callback=self.update_progress
            )

            if result["success"]:
                messagebox.showinfo("Success", f"Workflow '{workflow.name}' completed successfully!\nProcessed {len(result['processed_files'])} files.")
            else:
                messagebox.showerror("Workflow Failed", f"Workflow failed: {'; '.join(result['errors'])}")

        except Exception as e:
            messagebox.showerror("Error", f"Could not execute workflow: {str(e)}")

    def create_custom_workflow(self):
        """Create a new custom workflow"""
        try:
            # Simple dialog for workflow creation
            import tkinter.simpledialog as sd
            name = sd.askstring("Create Workflow", "Enter workflow name:")
            if not name:
                return

            description = sd.askstring("Create Workflow", "Enter workflow description:")
            if not description:
                return

            workflow_id = self.preset_manager.create_custom_workflow(name, description)
            messagebox.showinfo("Success", f"Workflow '{name}' created successfully!")

            # Refresh the GUI to show the new workflow
            self.refresh_workflow_buttons()

        except Exception as e:
            messagebox.showerror("Error", f"Could not create workflow: {str(e)}")

    def manage_workflows(self):
        """Open workflow management dialog"""
        try:
            # Create a simple workflow management window
            manage_window = tk.Toplevel(self.root)
            manage_window.title("Manage Custom Workflows")
            manage_window.geometry("600x400")

            # Listbox for workflows
            listbox = tk.Listbox(manage_window, height=10)
            listbox.pack(fill='both', expand=True, padx=10, pady=10)

            # Populate listbox
            workflows = self.preset_manager.get_all_custom_workflows()
            for workflow in workflows:
                listbox.insert(tk.END, f"{workflow.icon} {workflow.name} - {workflow.description}")

            # Buttons
            button_frame = ttk.Frame(manage_window)
            button_frame.pack(fill='x', padx=10, pady=5)

            ttk.Button(button_frame, text="Edit", command=lambda: self.edit_workflow(listbox, workflows)).pack(side='left', padx=5)
            ttk.Button(button_frame, text="Delete", command=lambda: self.delete_workflow(listbox, workflows, manage_window)).pack(side='left', padx=5)
            ttk.Button(button_frame, text="Close", command=manage_window.destroy).pack(side='right', padx=5)

        except Exception as e:
            messagebox.showerror("Error", f"Could not open workflow manager: {str(e)}")

    def show_smart_suggestions(self):
        """Show intelligent parameter suggestions based on current analysis"""
        try:
            if not self.current_recommendations:
                messagebox.showwarning("No Analysis", "Please analyze files first to get smart suggestions.")
                return

            # Get suggestions from preset manager
            suggestions = self.preset_manager.suggest_parameters({"content_analysis": self.current_recommendations})

            # Create suggestions window
            suggest_window = tk.Toplevel(self.root)
            suggest_window.title("Smart Parameter Suggestions")
            suggest_window.geometry("500x400")

            text_area = scrolledtext.ScrolledText(suggest_window, wrap=tk.WORD)
            text_area.pack(fill='both', expand=True, padx=10, pady=10)

            # Display suggestions
            text_area.insert(tk.END, "🎯 Smart Parameter Suggestions\n\n")
            text_area.insert(tk.END, f"Content Type: {suggestions['content_type']}\n")
            text_area.insert(tk.END, f"Recommended Bitrates: {', '.join(map(str, suggestions['bitrates']))}\n")
            text_area.insert(tk.END, f"Enable Compression: {suggestions['enable_compression']}\n")
            text_area.insert(tk.END, f"Enable Multiband: {suggestions['enable_multiband']}\n")
            text_area.insert(tk.END, f"Enable Loudness Norm: {suggestions['enable_loudnorm']}\n")
            text_area.insert(tk.END, f"Confidence: {suggestions['confidence']:.2f}\n\n")

            text_area.insert(tk.END, "Reasoning:\n")
            for reason in suggestions['reasoning']:
                text_area.insert(tk.END, f"• {reason}\n")

            text_area.config(state='disabled')

            # Apply button
            ttk.Button(suggest_window, text="Apply Suggestions",
                      command=lambda: self.apply_suggestions(suggestions, suggest_window)).pack(pady=10)

        except Exception as e:
            messagebox.showerror("Error", f"Could not show suggestions: {str(e)}")

    def apply_suggestions(self, suggestions, window):
        """Apply smart suggestions to settings"""
        try:
            self.format_var.set("mp3")  # Default format
            self.bitrates.set(','.join(map(str, suggestions['bitrates'])))
            self.content_type.set(suggestions['content_type'])
            self.normalize_var.set(suggestions['enable_loudnorm'])
            self.compressor_var.set(suggestions['enable_compression'])
            self.multiband_var.set(suggestions['enable_multiband'])

            messagebox.showinfo("Success", "Smart suggestions applied to settings!")
            window.destroy()

        except Exception as e:
            messagebox.showerror("Error", f"Could not apply suggestions: {str(e)}")

    def edit_workflow(self, listbox, workflows):
        """Edit selected workflow"""
        try:
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select a workflow to edit.")
                return

            workflow = workflows[selection[0]]
            # For now, just show info - full editing would need a more complex dialog
            messagebox.showinfo("Workflow Info", f"Name: {workflow.name}\nDescription: {workflow.description}\nSteps: {len(workflow.steps)}")

        except Exception as e:
            messagebox.showerror("Error", f"Could not edit workflow: {str(e)}")

    def delete_workflow(self, listbox, workflows, window):
        """Delete selected workflow"""
        try:
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select a workflow to delete.")
                return

            workflow = workflows[selection[0]]
            if messagebox.askyesno("Confirm Delete", f"Delete workflow '{workflow.name}'?"):
                self.preset_manager.delete_custom_workflow(workflow.id)
                messagebox.showinfo("Success", f"Workflow '{workflow.name}' deleted.")
                window.destroy()
                self.refresh_workflow_buttons()

        except Exception as e:
            messagebox.showerror("Error", f"Could not delete workflow: {str(e)}")

    def refresh_workflow_buttons(self):
        """Refresh workflow buttons in the main interface"""
        # This would need to be implemented to dynamically update the workflow buttons
        # For now, just show a message
        pass

    def update_progress(self, message):
        """Update progress display during workflow execution"""
        self.status_var.set(message)
        self.root.update_idletasks()

    def queue_with_preset(self):
        """Queue batch processing with a selected preset"""
        try:
            from presets import preset_manager
            from job_queue import job_queue

            # Get available presets
            presets = preset_manager.get_all_presets()
            if not presets:
                messagebox.showerror("Error", "No presets available!")
                return

            # Create preset selection dialog
            preset_names = [f"{p.icon} {p.name}" for p in presets]
            preset_dict = {f"{p.icon} {p.name}": p for p in presets}

            # Simple dialog for preset selection
            import tkinter.simpledialog as sd
            selected = sd.askstring("Select Preset", "Choose a preset for batch processing:",
                                  initialvalue=preset_names[0])

            if not selected or selected not in preset_dict:
                return

            preset = preset_dict[selected]

            # Validate inputs
            input_dir = self.input_dir.get()
            output_dir = self.output_dir.get()

            if not os.path.exists(input_dir):
                messagebox.showerror("Error", "Input directory does not exist!")
                return

            # Queue the batch processing
            job_ids = job_queue.add_preset_batch(input_dir, output_dir, preset.name)

            messagebox.showinfo("Success", f"Queued {len(job_ids)} jobs for preset '{preset.name}'")

            # Start job queue if not running
            job_queue.start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to queue batch processing: {str(e)}")

    def analyze_files(self):
        """Quick analysis of input files for recommendations"""
        try:
            input_dir = self.input_dir.get()
            if not os.path.exists(input_dir):
                messagebox.showerror("Error", "Input directory does not exist!")
                return

            # Switch to preview tab and run analysis
            self.notebook.select(2)  # Preview tab index
            self.generate_preview()

        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")

    def load_config(self):
        # Load settings from config
        try:
            defaults = self.config_manager.get_default_setting
            self.format_var.set(defaults('format') or 'mp3')
            self.content_type.set(defaults('content_type') or 'speech')
            self.normalize_var.set(defaults('loudnorm_enabled') if defaults('loudnorm_enabled') is not None else True)
            self.compressor_var.set(defaults('compressor_enabled') if defaults('compressor_enabled') is not None else False)
            self.multiband_var.set(defaults('multiband_enabled') if defaults('multiband_enabled') is not None else False)
            self.ml_noise_var.set(defaults('ml_noise_reduction') if defaults('ml_noise_reduction') is not None else False)
        except:
            pass  # Use defaults

    def save_config(self):
        # Save current settings to config
        try:
            self.config_manager.set_default_setting('format', self.format_var.get())
            self.config_manager.set_default_setting('content_type', self.content_type.get())
            self.config_manager.set_default_setting('loudnorm_enabled', self.normalize_var.get())
            self.config_manager.set_default_setting('compressor_enabled', self.compressor_var.get())
            self.config_manager.set_default_setting('multiband_enabled', self.multiband_var.get())
            self.config_manager.set_default_setting('ml_noise_reduction', self.ml_noise_var.get())

            # Validate configuration after saving
            is_valid, errors = self.config_manager.validate_config()
            if not is_valid:
                error_msg = "Configuration saved but has validation errors:\n" + "\n".join(errors)
                messagebox.showwarning("Configuration Warnings", error_msg)
            else:
                messagebox.showinfo("Success", "Settings saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")

def main():
    root = tk.Tk()
    app = AudioCompressorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()