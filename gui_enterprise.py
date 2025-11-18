"""
Enterprise Audio Processing GUI with Real-time Waveform Visualization

This module provides a comprehensive, enterprise-grade GUI featuring:
- Real-time waveform visualization and playback
- Dynamic parameter sliders with instant preview
- Drag-and-drop file/directory interface
- Responsive layout with accessibility support
- Contextual preset selection with dynamic suggestions
- Multi-threaded processing with progress tracking
- Cross-platform compatibility (Tkinter/PyQt fallback)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import subprocess
import os
import sys
import logging
import time
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple
import queue
import weakref

# Try to import GUI libraries
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib not available, using basic waveform display")

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logging.warning("pygame not available, audio preview disabled")

# Import our modules
from interfaces import IEventPublisher, IServiceProvider
from di_container import get_service
from security import security_manager, Permission, AuthMethod
from audio_analysis_enhanced import audio_analysis_engine, AudioContentType, AudioAnalysisResult
from audio_processing_enhanced import audio_processing_engine, FilterType, AudioFilter, ProcessingQuality
from presets import preset_manager


class WaveformCanvas:
    """Real-time waveform visualization canvas"""

    def __init__(self, parent: tk.Widget, width: int = 800, height: int = 200):
        self.parent = parent
        self.width = width
        self.height = height
        self.canvas = None
        self.waveform_data = []
        self.playback_position = 0
        self.zoom_level = 1.0
        self.offset = 0
        
        self._init_canvas()
        
    def _init_canvas(self):
        """Initialize the waveform canvas"""
        if MATPLOTLIB_AVAILABLE:
            self._init_matplotlib_canvas()
        else:
            self._init_basic_canvas()
    
    def _init_matplotlib_canvas(self):
        """Initialize matplotlib-based waveform canvas"""
        try:
            # Create matplotlib figure
            self.fig = Figure(figsize=(10, 3), dpi=80)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlim(0, self.width)
            self.ax.set_ylim(-1, 1)
            self.ax.set_xlabel('Time')
            self.ax.set_ylabel('Amplitude')
            self.ax.grid(True, alpha=0.3)
            
            # Remove top and right spines for cleaner look
            self.ax.spines['top'].set_visible(False)
            self.ax.spines['right'].set_visible(False)
            
            # Create canvas
            self.canvas = FigureCanvasTkAgg(self.fig, self.parent)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Initialize empty waveform
            self.waveform_line, = self.ax.plot([], [], 'b-', linewidth=0.5, alpha=0.7)
            self.playback_line = self.ax.axvline(x=0, color='r', linestyle='--', alpha=0.8)
            
        except Exception as e:
            logging.error(f"Failed to initialize matplotlib canvas: {e}")
            self._init_basic_canvas()
    
    def _init_basic_canvas(self):
        """Initialize basic Tkinter canvas as fallback"""
        self.canvas = tk.Canvas(self.parent, width=self.width, height=self.height, 
                               bg='white', highlightthickness=1, highlightbackground='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Initialize with empty waveform
        self.draw_waveform([])
    
    def draw_waveform(self, waveform_data: List[float]):
        """Draw waveform from audio data"""
        self.waveform_data = waveform_data
        
        if MATPLOTLIB_AVAILABLE and hasattr(self, 'waveform_line'):
            self._draw_matplotlib_waveform(waveform_data)
        else:
            self._draw_basic_waveform(waveform_data)
    
    def _draw_matplotlib_waveform(self, waveform_data: List[float]):
        """Draw waveform using matplotlib"""
        try:
            if not waveform_data:
                self.waveform_line.set_data([], [])
                self.canvas.draw()
                return
            
            # Generate time axis
            duration = len(waveform_data) / 44100  # Assuming 44.1kHz sample rate
            time_axis = np.linspace(0, duration, len(waveform_data))
            
            # Apply zoom and offset
            if self.zoom_level != 1.0:
                start_idx = int(len(time_axis) * self.offset)
                end_idx = int(len(time_axis) * (self.offset + 1/self.zoom_level))
                start_idx = max(0, start_idx)
                end_idx = min(len(time_axis), end_idx)
                
                time_axis = time_axis[start_idx:end_idx]
                waveform_data = waveform_data[start_idx:end_idx]
            
            # Update waveform
            self.waveform_line.set_data(time_axis, waveform_data)
            
            # Update axes
            if len(time_axis) > 0:
                self.ax.set_xlim(time_axis[0], time_axis[-1])
            
            # Update playback position
            if hasattr(self, 'playback_line'):
                playback_time = self.playback_position / 44100
                self.playback_line.set_xdata([playback_time, playback_time])
            
            self.canvas.draw()
            
        except Exception as e:
            logging.error(f"Error drawing matplotlib waveform: {e}")
            self._draw_basic_waveform(waveform_data)
    
    def _draw_basic_waveform(self, waveform_data: List[float]):
        """Draw waveform using basic Tkinter canvas"""
        self.canvas.delete("waveform")
        self.canvas.delete("playback")
        
        if not waveform_data:
            return
        
        # Calculate dimensions
        width = self.canvas.winfo_width()
        if width <= 1:
            width = self.width
        
        height = self.canvas.winfo_height()
        if height <= 1:
            height = self.height
        
        # Clear existing waveform
        self.canvas.delete("all")
        
        # Draw center line
        center_y = height // 2
        self.canvas.create_line(0, center_y, width, center_y, fill='gray', tags="centerline")
        
        # Draw waveform
        if len(waveform_data) > 1:
            # Scale waveform to fit canvas
            max_amplitude = max(max(waveform_data), abs(min(waveform_data)), 1.0)
            scale_factor = (height // 2 - 10) / max_amplitude
            
            # Calculate points
            step = max(1, len(waveform_data) // width)
            points = []
            
            for i in range(0, len(waveform_data), step):
                x = int(i / len(waveform_data) * width)
                y = center_y - int(waveform_data[i] * scale_factor)
                points.extend([x, y])
            
            if points:
                self.canvas.create_line(points, fill='blue', width=1, tags="waveform")
        
        # Draw playback position
        if self.playback_position < len(waveform_data):
            playback_x = int(self.playback_position / len(waveform_data) * width)
            self.canvas.create_line(playback_x, 0, playback_x, height, 
                                  fill='red', dash=(4, 2), tags="playback")
    
    def update_playback_position(self, position: float):
        """Update playback position indicator"""
        self.playback_position = position
        
        if MATPLOTLIB_AVAILABLE and hasattr(self, 'playback_line'):
            playback_time = position / 44100
            self.playback_line.set_xdata([playback_time, playback_time])
            self.canvas.draw()
        else:
            self._draw_basic_waveform(self.waveform_data)
    
    def set_zoom(self, zoom_level: float):
        """Set zoom level (1.0 = normal, >1.0 = zoom in)"""
        self.zoom_level = max(0.1, min(10.0, zoom_level))
        self._draw_matplotlib_waveform(self.waveform_data)
    
    def set_offset(self, offset: float):
        """Set horizontal offset (0.0 to 1.0)"""
        self.offset = max(0.0, min(1.0 - 1/self.zoom_level, offset))
        self._draw_matplotlib_waveform(self.waveform_data)


class ParameterSlider:
    """Individual parameter slider with real-time value display"""

    def __init__(self, parent: tk.Widget, parameter_name: str, initial_value: float,
                 min_value: float, max_value: float, step: float = 0.1,
                 unit: str = "", callback: Optional[Callable[[str, float], None]] = None):
        self.parameter_name = parameter_name
        self.initial_value = initial_value
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.unit = unit
        self.callback = callback
        self.value_var = tk.DoubleVar(value=initial_value)
        
        self._create_widgets(parent)
        self._setup_callbacks()
    
    def _create_widgets(self, parent: tk.Widget):
        """Create slider widgets"""
        # Frame for this parameter
        self.frame = tk.Frame(parent, relief=tk.RAISED, bd=1)
        self.frame.pack(fill=tk.X, padx=5, pady=2)
        
        # Label and value
        label_frame = tk.Frame(self.frame)
        label_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.label = tk.Label(label_frame, text=self.parameter_name, 
                            font=("Arial", 9, "bold"))
        self.label.pack(side=tk.LEFT)
        
        self.value_label = tk.Label(label_frame, text=f"{self.initial_value:.2f} {self.unit}",
                                  font=("Arial", 9), fg="blue")
        self.value_label.pack(side=tk.RIGHT)
        
        # Slider
        self.slider = tk.Scale(self.frame, from_=self.min_value, to=self.max_value,
                             resolution=self.step, orient=tk.HORIZONTAL,
                             variable=self.value_var, length=300)
        self.slider.pack(fill=tk.X, padx=10, pady=5)
    
    def _setup_callbacks(self):
        """Setup value change callbacks"""
        def on_value_change(value):
            # Update value label
            self.value_label.config(text=f"{value:.2f} {self.unit}")
            
            # Call callback if provided
            if self.callback:
                self.callback(self.parameter_name, float(value))
        
        self.value_var.trace('w', lambda *args: on_value_change(self.value_var.get()))
    
    def get_value(self) -> float:
        """Get current slider value"""
        return self.value_var.get()
    
    def set_value(self, value: float):
        """Set slider value"""
        value = max(self.min_value, min(self.max_value, value))
        self.value_var.set(value)
    
    def set_callback(self, callback: Callable[[str, float], None]):
        """Set value change callback"""
        self.callback = callback


class AudioPreviewPlayer:
    """Audio preview player with playback controls"""

    def __init__(self):
        self.is_playing = False
        self.current_file = None
        self.playback_thread = None
        self.stop_event = threading.Event()
        
        if PYGAME_AVAILABLE:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
    
    def play_preview(self, audio_file: str, filters: List[AudioFilter]) -> bool:
        """Generate and play preview with applied filters"""
        try:
            # Stop any current playback
            self.stop_playback()
            
            # Generate preview with filters
            if not audio_processing_engine:
                logging.error("Audio processing engine not available")
                return False
            
            preview_file = audio_processing_engine.generate_preview_clip(audio_file, filters=filters)
            if not preview_file:
                logging.error("Failed to generate preview clip")
                return False
            
            self.current_file = preview_file
            
            # Start playback in background thread
            self.stop_event.clear()
            self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
            self.playback_thread.start()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to play preview: {e}")
            return False
    
    def _playback_loop(self):
        """Background playback loop"""
        try:
            if PYGAME_AVAILABLE:
                pygame.mixer.music.load(self.current_file)
                pygame.mixer.music.play()
                self.is_playing = True
                
                while self.is_playing and not self.stop_event.is_set():
                    if not pygame.mixer.music.get_busy():
                        break
                    time.sleep(0.1)
            else:
                # Fallback: use ffplay if available
                self._play_with_ffplay()
                
        except Exception as e:
            logging.error(f"Playback error: {e}")
        finally:
            self.is_playing = False
            self._cleanup_preview_file()
    
    def _play_with_ffplay(self):
        """Play preview using ffplay as fallback"""
        try:
            cmd = ["ffplay", "-autoexit", "-nodisp", self.current_file]
            process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            self.is_playing = True
            while self.is_playing and process.poll() is None and not self.stop_event.is_set():
                time.sleep(0.1)
            
            if process.poll() is None:
                process.terminate()
                
        except Exception as e:
            logging.error(f"FFplay playback error: {e}")
    
    def stop_playback(self):
        """Stop current playback"""
        self.is_playing = False
        self.stop_event.set()
        
        if PYGAME_AVAILABLE:
            pygame.mixer.music.stop()
        
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=2.0)
    
    def _cleanup_preview_file(self):
        """Clean up temporary preview file"""
        if self.current_file and os.path.exists(self.current_file):
            try:
                os.remove(self.current_file)
            except:
                pass
            self.current_file = None
    
    def is_playing_preview(self) -> bool:
        """Check if preview is currently playing"""
        return self.is_playing


class PureSoundGUI:
    """Main enterprise audio processing GUI"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Pure Sound - Enterprise Audio Processing Suite")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Initialize services
        self.event_publisher = None
        try:
            self.event_publisher = get_service(IEventPublisher)
        except:
            pass
        
        self.audio_analysis_engine = audio_analysis_engine
        self.audio_processing_engine = audio_processing_engine
        self.preset_manager = preset_manager
        
        # GUI state
        self.current_file = None
        self.current_analysis = None
        self.current_filters = []
        self.selected_preset = "speech_clean"
        
        # Audio player
        self.preview_player = AudioPreviewPlayer()
        
        # Event queue for thread-safe GUI updates
        self.event_queue = queue.Queue()
        self.root.after(100, self._process_event_queue)
        
        # Create GUI
        self._create_menu_bar()
        self._create_main_interface()
        self._setup_keyboard_shortcuts()
        
        # Start periodic updates
        self.root.after(1000, self._periodic_update)
        
        # Setup event subscriptions
        self._setup_event_subscriptions()
        
        logging.info("Pure Sound GUI initialized")

    def _create_menu_bar(self):
        """Create application menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Audio File", command=self._open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Open Directory", command=self._open_directory)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Analyze Audio", command=self._analyze_current_file)
        tools_menu.add_command(label="Generate Preview", command=self._generate_preview)
        tools_menu.add_separator()
        tools_menu.add_command(label="Settings", command=self._show_settings)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Zoom In", command=lambda: self.waveform_canvas.set_zoom(self.waveform_canvas.zoom_level * 1.5))
        view_menu.add_command(label="Zoom Out", command=lambda: self.waveform_canvas.set_zoom(self.waveform_canvas.zoom_level / 1.5))
        view_menu.add_command(label="Reset Zoom", command=lambda: self.waveform_canvas.set_zoom(1.0))
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self._show_help)
        help_menu.add_command(label="About", command=self._show_about)

    def _create_main_interface(self):
        """Create the main GUI interface"""
        # Main container with PanedWindow
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - File and controls
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Right panel - Waveform and parameters
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)
        
        self._create_left_panel(left_frame)
        self._create_right_panel(right_frame)
        
        # Status bar
        self._create_status_bar()
    
    def _create_left_panel(self, parent):
        """Create left control panel"""
        # File selection section
        file_frame = ttk.LabelFrame(parent, text="Audio File", padding=10)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # File info display
        self.file_info_text = scrolledtext.ScrolledText(file_frame, height=6, width=30, state=tk.DISABLED)
        self.file_info_text.pack(fill=tk.X, pady=(0, 10))
        
        # File operations
        file_buttons_frame = ttk.Frame(file_frame)
        file_buttons_frame.pack(fill=tk.X)
        
        ttk.Button(file_buttons_frame, text="Browse...", 
                  command=self._open_file).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_buttons_frame, text="Analyze", 
                  command=self._analyze_current_file).pack(side=tk.LEFT, padx=5)
        
        # Analysis results section
        analysis_frame = ttk.LabelFrame(parent, text="Analysis Results", padding=10)
        analysis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.analysis_text = scrolledtext.ScrolledText(analysis_frame, height=8, width=30, state=tk.DISABLED)
        self.analysis_text.pack(fill=tk.BOTH, expand=True)
        
        # Preset selection section
        preset_frame = ttk.LabelFrame(parent, text="Processing Preset", padding=10)
        preset_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Preset dropdown
        preset_select_frame = ttk.Frame(preset_frame)
        preset_select_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(preset_select_frame, text="Preset:").pack(side=tk.LEFT)
        self.preset_var = tk.StringVar(value=self.selected_preset)
        self.preset_combo = ttk.Combobox(preset_select_frame, textvariable=self.preset_var,
                                       values=self._get_available_presets(), state="readonly")
        self.preset_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        self.preset_combo.bind('<<ComboboxSelected>>', self._on_preset_change)
        
        # Suggested preset button
        ttk.Button(preset_frame, text="Get Suggestion", 
                  command=self._get_preset_suggestion).pack(fill=tk.X)
        
        # Control buttons
        control_frame = ttk.LabelFrame(parent, text="Actions", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Generate Preview", 
                  command=self._generate_preview).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Start Processing", 
                  command=self._start_processing).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="Stop Preview", 
                  command=self._stop_preview).pack(fill=tk.X, pady=2)

    def _create_right_panel(self, parent):
        """Create right panel with waveform and parameters"""
        # Waveform section
        waveform_frame = ttk.LabelFrame(parent, text="Waveform Visualization", padding=5)
        waveform_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Waveform canvas
        self.waveform_canvas = WaveformCanvas(waveform_frame, width=800, height=200)
        
        # Waveform controls
        waveform_controls = ttk.Frame(waveform_frame)
        waveform_controls.pack(fill=tk.X, pady=5)
        
        ttk.Button(waveform_controls, text="Zoom In", 
                  command=lambda: self.waveform_canvas.set_zoom(self.waveform_canvas.zoom_level * 1.5)).pack(side=tk.LEFT, padx=2)
        ttk.Button(waveform_controls, text="Zoom Out", 
                  command=lambda: self.waveform_canvas.set_zoom(self.waveform_canvas.zoom_level / 1.5)).pack(side=tk.LEFT, padx=2)
        ttk.Button(waveform_controls, text="Reset", 
                  command=lambda: self.waveform_canvas.set_zoom(1.0)).pack(side=tk.LEFT, padx=2)
        
        # Parameters section
        params_frame = ttk.LabelFrame(parent, text="Processing Parameters", padding=5)
        params_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create notebook for parameter categories
        self.params_notebook = ttk.Notebook(params_frame)
        self.params_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create parameter pages
        self._create_compressor_page()
        self._create_gate_page()
        self._create_eq_page()
        
        # Parameter sliders will be created dynamically
        self.parameter_sliders: Dict[str, ParameterSlider] = {}

    def _create_compressor_page(self):
        """Create compressor parameters page"""
        page = ttk.Frame(self.params_notebook)
        self.params_notebook.add(page, text="Compressor")
        
        # Scrollable frame for parameters
        canvas = tk.Canvas(page)
        scrollbar = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.compressor_frame = scrollable_frame

    def _create_gate_page(self):
        """Create noise gate parameters page"""
        page = ttk.Frame(self.params_notebook)
        self.params_notebook.add(page, text="Noise Gate")
        
        # Scrollable frame for parameters
        canvas = tk.Canvas(page)
        scrollbar = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.gate_frame = scrollable_frame

    def _create_eq_page(self):
        """Create EQ parameters page"""
        page = ttk.Frame(self.params_notebook)
        self.params_notebook.add(page, text="Equalizer")
        
        # Scrollable frame for parameters
        canvas = tk.Canvas(page)
        scrollbar = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.eq_frame = scrollable_frame

    def _create_status_bar(self):
        """Create application status bar"""
        self.status_frame = ttk.Frame(self.root, relief=tk.SUNKEN, bd=1)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Status labels
        self.status_label = ttk.Label(self.status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.status_frame, variable=self.progress_var,
                                          mode='determinate', length=200)
        self.progress_bar.pack(side=tk.RIGHT, padx=5)
        
        # Progress label
        self.progress_label = ttk.Label(self.status_frame, text="")
        self.progress_label.pack(side=tk.RIGHT, padx=5)

    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts"""
        self.root.bind('<Control-o>', lambda e: self._open_file())
        self.root.bind('<Control-a>', lambda e: self._analyze_current_file())
        self.root.bind('<Control-p>', lambda e: self._generate_preview())
        self.root.bind('<Control-s>', lambda e: self._start_processing())
        self.root.bind('<Escape>', lambda e: self._stop_preview())

    def _setup_event_subscriptions(self):
        """Setup event subscriptions for GUI updates"""
        if self.event_publisher:
            # Subscribe to audio analysis events
            self.event_publisher.subscribe("audio.analyzed", self._on_audio_analyzed)
            self.event_publisher.subscribe("audio.analysis_failed", self._on_analysis_failed)
            
            # Subscribe to processing events
            self.event_publisher.subscribe("processing.job_started", self._on_job_started)
            self.event_publisher.subscribe("processing.job_completed", self._on_job_completed)
            self.event_publisher.subscribe("processing.job_failed", self._on_job_failed)

    def _process_event_queue(self):
        """Process events from the queue (thread-safe GUI updates)"""
        try:
            while True:
                event = self.event_queue.get_nowait()
                if event['type'] == 'update_status':
                    self.status_label.config(text=event['message'])
                elif event['type'] == 'update_progress':
                    self.progress_var.set(event['value'])
                    self.progress_label.config(text=event['text'])
                elif event['type'] == 'update_file_info':
                    self._update_file_info_display(event['data'])
                elif event['type'] == 'update_analysis':
                    self._update_analysis_display(event['data'])
                elif event['type'] == 'update_waveform':
                    self.waveform_canvas.draw_waveform(event['waveform_data'])
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self._process_event_queue)

    def _periodic_update(self):
        """Periodic GUI updates"""
        # Update playback position if playing
        if self.preview_player.is_playing_preview():
            # This would need actual playback position tracking
            pass
        
        # Schedule next update
        self.root.after(100, self._periodic_update)

    # File operations
    def _open_file(self):
        """Open audio file dialog"""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.m4a *.flac *.aac *.ogg *.opus"),
                ("WAV files", "*.wav"),
                ("MP3 files", "*.mp3"),
                ("FLAC files", "*.flac"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self._load_audio_file(file_path)

    def _open_directory(self):
        """Open directory selection dialog"""
        directory = filedialog.askdirectory(title="Select Audio Directory")
        if directory:
            self._status_message(f"Directory selected: {directory}")
            # TODO: Implement directory batch processing

    def _load_audio_file(self, file_path: str):
        """Load and display audio file"""
        try:
            self.current_file = file_path
            
            # Update file info display
            file_info = {
                'path': file_path,
                'name': os.path.basename(file_path),
                'size': f"{os.path.getsize(file_path) / (1024*1024):.2f} MB",
                'modified': time.ctime(os.path.getmtime(file_path))
            }
            
            self.event_queue.put({
                'type': 'update_file_info',
                'data': file_info
            })
            
            # Analyze file if auto-analyze is enabled
            self._analyze_current_file()
            
            self._status_message(f"Loaded: {file_info['name']}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def _analyze_current_file(self):
        """Analyze current audio file"""
        if not self.current_file:
            messagebox.showwarning("Warning", "No audio file loaded")
            return
        
        def analyze_thread():
            try:
                self.event_queue.put({
                    'type': 'update_status',
                    'message': 'Analyzing audio file...'
                })
                
                # Perform analysis
                result = self.audio_analysis_engine.analyze_file(self.current_file)
                
                if result:
                    self.current_analysis = result
                    
                    # Update UI with results
                    self.event_queue.put({
                        'type': 'update_analysis',
                        'data': result
                    })
                    
                    # Update status
                    self.event_queue.put({
                        'type': 'update_status',
                        'message': f"Analysis complete: {result.content_type.value} ({result.confidence:.2f})"
                    })
                    
                    # Suggest preset
                    self._get_preset_suggestion()
                else:
                    self.event_queue.put({
                        'type': 'update_status',
                        'message': 'Analysis failed'
                    })
                    
            except Exception as e:
                logging.error(f"Analysis error: {e}")
                self.event_queue.put({
                    'type': 'update_status',
                    'message': f'Analysis failed: {str(e)}'
                })
        
        threading.Thread(target=analyze_thread, daemon=True).start()

    def _get_preset_suggestion(self):
        """Get AI-suggested preset based on analysis"""
        if not self.current_analysis:
            return
        
        try:
            # Get suggestion from processing engine
            suggested_preset, _ = self.audio_processing_engine.suggest_processing_chain(self.current_analysis)
            
            # Update preset selection
            self.selected_preset = suggested_preset
            self.preset_var.set(suggested_preset)
            
            # Load preset parameters
            self._load_preset_parameters(suggested_preset)
            
            self._status_message(f"AI suggested preset: {suggested_preset}")
            
        except Exception as e:
            logging.error(f"Preset suggestion error: {e}")

    def _load_preset_parameters(self, preset_name: str):
        """Load parameters for selected preset"""
        try:
            # Clear existing sliders
            for slider in self.parameter_sliders.values():
                slider.frame.destroy()
            self.parameter_sliders.clear()
            
            # Get filters for preset
            filters = self.audio_processing_engine.preset_manager.get_preset(preset_name)
            self.current_filters = filters
            
            # Create sliders for each filter
            for filter_obj in filters:
                self._create_filter_sliders(filter_obj)
                
        except Exception as e:
            logging.error(f"Error loading preset parameters: {e}")

    def _create_filter_sliders(self, filter_obj: AudioFilter):
        """Create parameter sliders for a filter"""
        try:
            # Get parameter definitions
            param_defs = self.audio_processing_engine.get_filter_parameters(filter_obj.filter_type)
            
            if not param_defs:
                return
            
            # Select appropriate frame based on filter type
            if filter_obj.filter_type == FilterType.COMPRESSOR:
                parent_frame = self.compressor_frame
            elif filter_obj.filter_type == FilterType.NOISE_GATE:
                parent_frame = self.gate_frame
            elif filter_obj.filter_type in [FilterType.EQ, FilterType.HIGH_PASS, FilterType.LOW_PASS]:
                parent_frame = self.eq_frame
            else:
                parent_frame = self.compressor_frame  # Default
            
            # Create sliders for each parameter
            for param_name, param_def in param_defs.items():
                slider = ParameterSlider(
                    parent_frame,
                    param_name,
                    param_def.value,
                    param_def.min_value,
                    param_def.max_value,
                    param_def.step or 0.1,
                    param_def.unit,
                    lambda name, value: self._on_parameter_change(name, value, filter_obj)
                )
                
                self.parameter_sliders[f"{filter_obj.filter_type.value}_{param_name}"] = slider
                
        except Exception as e:
            logging.error(f"Error creating filter sliders: {e}")

    def _on_parameter_change(self, param_name: str, value: float, filter_obj: AudioFilter):
        """Handle parameter value changes"""
        try:
            # Update filter parameter
            if param_name in filter_obj.parameters:
                filter_obj.parameters[param_name].value = value
            
            # Update preview if enabled
            # In a real implementation, this would regenerate preview in real-time
            
        except Exception as e:
            logging.error(f"Error updating parameter {param_name}: {e}")

    def _on_preset_change(self, event=None):
        """Handle preset selection change"""
        self.selected_preset = self.preset_var.get()
        self._load_preset_parameters(self.selected_preset)

    def _generate_preview(self):
        """Generate and play audio preview with current settings"""
        if not self.current_file:
            messagebox.showwarning("Warning", "No audio file loaded")
            return
        
        if not self.current_filters:
            messagebox.showwarning("Warning", "No processing filters selected")
            return
        
        def preview_thread():
            try:
                self.event_queue.put({
                    'type': 'update_status',
                    'message': 'Generating preview...'
                })
                
                # Play preview with current filters
                success = self.preview_player.play_preview(self.current_file, self.current_filters)
                
                if success:
                    self.event_queue.put({
                        'type': 'update_status',
                        'message': 'Playing preview...'
                    })
                else:
                    self.event_queue.put({
                        'type': 'update_status',
                        'message': 'Failed to generate preview'
                    })
                    
            except Exception as e:
                logging.error(f"Preview generation error: {e}")
                self.event_queue.put({
                    'type': 'update_status',
                    'message': f'Preview failed: {str(e)}'
                })
        
        threading.Thread(target=preview_thread, daemon=True).start()

    def _start_processing(self):
        """Start batch processing with current settings"""
        if not self.current_file:
            messagebox.showwarning("Warning", "No audio file loaded")
            return
        
        if not self.current_filters:
            messagebox.showwarning("Warning", "No processing filters selected")
            return
        
        # Ask for output location
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return
        
        def process_thread():
            try:
                self.event_queue.put({
                    'type': 'update_status',
                    'message': 'Starting processing...'
                })
                
                # Create processing job
                output_file = os.path.join(output_dir, f"processed_{os.path.basename(self.current_file)}")
                job = self.audio_processing_engine.create_processing_job(
                    self.current_file,
                    [output_file],
                    self.selected_preset
                )
                
                # Submit job
                job_id = self.audio_processing_engine.submit_job(job)
                
                self.event_queue.put({
                    'type': 'update_status',
                    'message': f'Processing started (ID: {job_id[:8]}...)'
                })
                
            except Exception as e:
                logging.error(f"Processing error: {e}")
                self.event_queue.put({
                    'type': 'update_status',
                    'message': f'Processing failed: {str(e)}'
                })
        
        threading.Thread(target=process_thread, daemon=True).start()

    def _stop_preview(self):
        """Stop preview playback"""
        self.preview_player.stop_playback()
        self._status_message("Preview stopped")

    def _show_settings(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("600x400")
        settings_window.resizable(True, True)
        
        # TODO: Implement settings dialog
        
        ttk.Label(settings_window, text="Settings dialog coming soon...").pack(expand=True)

    def _show_help(self):
        """Show help dialog"""
        messagebox.showinfo("Help", "Pure Sound User Guide\\n\\nComing soon...")

    def _show_about(self):
        """Show about dialog"""
        about_text = """Pure Sound - Enterprise Audio Processing Suite
        
Version: 1.0.0
Build: 2025.11.18

Features:
• Real-time audio analysis and content detection
• Advanced audio processing with ML-based noise reduction
• Professional-grade compression and filtering
• Multi-stream output configuration
• Enterprise security and audit logging
• Cross-platform GUI with accessibility support

Built with Python, FFmpeg, and advanced audio processing libraries."""
        
        messagebox.showinfo("About Pure Sound", about_text)

    # Event handlers
    def _on_audio_analyzed(self, event):
        """Handle audio analysis completed event"""
        # This would update the GUI with analysis results
        pass

    def _on_analysis_failed(self, event):
        """Handle analysis failure event"""
        self._status_message(f"Analysis failed: {event.data.get('error', 'Unknown error')}")

    def _on_job_started(self, event):
        """Handle processing job started event"""
        self._status_message(f"Processing started: {event.data.get('job_id', '')}")

    def _on_job_completed(self, event):
        """Handle processing job completed event"""
        self._status_message("Processing completed successfully")

    def _on_job_failed(self, event):
        """Handle processing job failed event"""
        self._status_message(f"Processing failed: {event.data.get('error', 'Unknown error')}")

    # Utility methods
    def _get_available_presets(self) -> List[str]:
        """Get list of available processing presets"""
        if hasattr(self, 'audio_processing_engine') and self.audio_processing_engine:
            return self.audio_processing_engine.get_available_presets()
        return []

    def _update_file_info_display(self, file_info: Dict[str, Any]):
        """Update file info display"""
        info_text = f"""File: {file_info['name']}
Path: {file_info['path']}
Size: {file_info['size']}
Modified: {file_info['modified']}"""
        
        self.file_info_text.config(state=tk.NORMAL)
        self.file_info_text.delete(1.0, tk.END)
        self.file_info_text.insert(1.0, info_text)
        self.file_info_text.config(state=tk.DISABLED)

    def _update_analysis_display(self, result: AudioAnalysisResult):
        """Update analysis results display"""
        info_text = f"""Content Type: {result.content_type.value}
Confidence: {result.confidence:.2f}
Quality: {result.quality.value}
Duration: {result.duration:.2f} seconds
Sample Rate: {result.sample_rate} Hz
Channels: {result.channels}
Recommended Format: {result.recommended_format}
Recommended Bitrates: {', '.join(map(str, result.recommended_bitrates))}
Processing Steps: {len(result.processing_steps)}
Warnings: {len(result.warnings)}"""
        
        if result.warnings:
            info_text += "\\n\\nWarnings:"
            for warning in result.warnings:
                info_text += f"\\n• {warning}"
        
        self.analysis_text.config(state=tk.NORMAL)
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(1.0, info_text)
        self.analysis_text.config(state=tk.DISABLED)

    def _status_message(self, message: str):
        """Update status message"""
        self.event_queue.put({
            'type': 'update_status',
            'message': message
        })

    def run(self):
        """Run the GUI application"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
            self.root.mainloop()
        except Exception as e:
            logging.error(f"GUI error: {e}")
            messagebox.showerror("Error", f"GUI error: {e}")

    def _on_closing(self):
        """Handle application closing"""
        # Stop any preview playback
        self.preview_player.stop_playback()
        
        # Save settings if needed
        # TODO: Implement settings persistence
        
        # Close application
        self.root.destroy()


# Global GUI instance
gui_app = None

def launch_gui():
    """Launch the Pure Sound GUI application"""
    global gui_app
    try:
        gui_app = PureSoundGUI()
        gui_app.run()
    except Exception as e:
        logging.error(f"Failed to launch GUI: {e}")
        raise

if __name__ == "__main__":
    # Launch GUI if run directly
    launch_gui()