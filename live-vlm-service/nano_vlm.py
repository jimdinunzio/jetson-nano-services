#!/usr/bin/env python3
"""
NanoVlm - A class wrapper for the nano_llm vision example.

This module provides a threaded, controllable interface for running
vision-language model inference on video streams.
"""

import time
import threading
from collections import deque

from nano_llm import NanoLLM, ChatHistory
from nano_llm.utils import ArgParser, load_prompts
from nano_llm.plugins import VideoSource

class NanoVlm:
    """
    A class that encapsulates vision-language model processing on video streams.
    
    Designed to be run in a separate thread with startUp() as the entry point.
    Provides thread-safe methods for controlling prompts and retrieving outputs.
    
    Supports enable/disable for low power mode - when disabled, no frames are
    captured or processed. Auto-disables after 30s of no get_output() calls.
    """
    
    # Auto-disable timeout in seconds
    AUTO_DISABLE_TIMEOUT = 30.0
    
    def __init__(self,
                 model="Efficient-Large-Model/VILA1.5-3b",
                 api=None,
                 quantization=None,
                 max_context_len=256,
                 vision_model=None,
                 vision_scaling=None,
                 chat_template=None,
                 system_prompt=None,
                 video_input="/dev/video0",
                 max_new_tokens=32,
                 min_new_tokens=None,
                 do_sample=None,
                 repetition_penalty=None,
                 temperature=None,
                 top_p=None,
                 prompts=None,
                 output_stack_size=25):
        """
        Initialize NanoVlm with configuration parameters.
        
        Args:
            model: Model path or name (default: "Efficient-Large-Model/VILA1.5-3b")
            api: API type for model loading
            quantization: Quantization settings
            max_context_len: Maximum context length
            vision_model: Vision model settings
            vision_scaling: Vision scaling settings
            chat_template: Chat template to use
            system_prompt: System prompt for the model
            video_input: Video input source (file pattern, device, or stream)
            max_new_tokens: Maximum new tokens to generate
            min_new_tokens: Minimum new tokens to generate
            do_sample: Whether to use sampling
            repetition_penalty: Repetition penalty value
            temperature: Temperature for generation
            top_p: Top-p (nucleus) sampling parameter
            prompts: List of prompts to apply to each frame
            output_stack_size: Size of the output history stack (default: 25)
        """
        # Model configuration
        self.model_name = model
        self.api = api
        self.quantization = quantization
        self.max_context_len = max_context_len
        self.vision_model = vision_model
        self.vision_scaling = vision_scaling
        
        # Chat configuration
        self.chat_template = chat_template
        self.system_prompt = system_prompt
        
        # Video configuration
        self.video_input = video_input
        
        # Generation parameters
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.top_p = top_p
        
        # Prompts - use defaults if not provided
        if prompts is None:
            self._prompts = ["Describe the image.", "Are there people in the image?"]
        else:
            self._prompts = list(prompts)
        
        # Output stack configuration
        self._output_stack_size = output_stack_size
        self._output_stack = deque(maxlen=output_stack_size)
        
        # Threading control
        self._run_flag = False
        self._lock = threading.Lock()
        
        # Enable/disable control for low power mode
        # Starts disabled - must call enable() to start processing
        self._enabled = False
        self._enabled_event = threading.Event()  # For fast wake-up
        
        # Auto-disable tracking
        self._last_get_output_time = time.time()
        
        # Model and video source (initialized in startUp)
        self._model = None
        self._chat_history = None
        self._video_source = None
    
    def enable(self):
        """
        Enable frame processing (exit low power mode).
        
        When enabled, frames will be captured, embedded, and text generated.
        """
        with self._lock:
            self._enabled = True
            self._last_get_output_time = time.time()  # Reset auto-disable timer
        self._enabled_event.set()  # Wake up the main loop immediately
    
    def disable(self):
        """
        Disable frame processing (enter low power mode).
        
        When disabled, no frames are captured or processed, reducing power consumption.
        Clears prompts and output stack. Call set_prompts() to re-enable with new prompts.
        """
        with self._lock:
            self._enabled = False
            self._prompts = []  # Clear prompts
            self._output_stack.clear()
        self._enabled_event.clear()
    
    def is_enabled(self):
        """Check if frame processing is currently enabled."""
        with self._lock:
            return self._enabled
    
    def _check_auto_disable(self):
        """
        Check if we should auto-disable due to no recent get_output() calls.
        
        Returns:
            True if auto-disabled, False otherwise
        """
        with self._lock:
            if not self._enabled:
                return False
            
            elapsed = time.time() - self._last_get_output_time
            if elapsed >= self.AUTO_DISABLE_TIMEOUT:
                self._enabled = False
                self._enabled_event.clear()
                self._prompts = []  # Clear prompts
                self._output_stack.clear()
                print(f"[NanoVlm] Auto-disabled after {elapsed:.1f}s of no get_output() calls")
                return True
            return False
    
    def startUp(self):
        """
        Main processing loop - designed to be run in a separate thread.
        
        Loads the model, opens the video source, and continuously processes
        frames with the configured prompts until shutDown() is called or
        the video source reaches end-of-stream.
        
        Starts in disabled (low power) mode. Call enable() to start processing.
        """
        self._run_flag = True
        
        # Load vision/language model
        model_kwargs = {
            'api': self.api,
            'quantization': self.quantization,
            'vision_model': self.vision_model,
            'vision_scaling': self.vision_scaling,
        }
        # Only include non-None values
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
        if self.max_context_len is not None:
            model_kwargs['max_context_len'] = self.max_context_len
        
        self._model = NanoLLM.from_pretrained(self.model_name, **model_kwargs)
        
        assert self._model.has_vision, "Model must have vision capabilities"
        
        # Create chat history
        self._chat_history = ChatHistory(
            self._model, 
            self.chat_template, 
            self.system_prompt
        )
        
        # Open video source
        video_kwargs = {
            'video_input': self.video_input,
            'cuda_stream': 0,
            'return_copy': False,
        }
        self._video_source = VideoSource(**video_kwargs)
        
        print("[NanoVlm] Model loaded. Starting in disabled (low power) mode.")
        print("[NanoVlm] Call enable() to start processing frames.")
        
        # Main processing loop
        while self._run_flag:
            # Check if we should auto-disable
            self._check_auto_disable()
            
            # Wait for enabled state (with timeout for checking run_flag and auto-disable)
            with self._lock:
                enabled = self._enabled
            
            if not enabled:
                # Wait for enable signal or timeout (check every 0.5s for shutdown)
                self._enabled_event.wait(timeout=0.5)
                continue
            
            img = self._video_source.capture()
            
            if img is None:
                continue
            
            self._chat_history.append('user', image=img)
            time_begin = time.perf_counter()
            
            # Get current prompts (thread-safe)
            with self._lock:
                current_prompts = list(self._prompts)
                # Double-check still enabled
                if not self._enabled:
                    self._chat_history.reset()
                    continue
            
            for prompt in current_prompts:
                if not self._run_flag:
                    break
                
                # Check if disabled mid-processing
                with self._lock:
                    if not self._enabled:
                        break
                    
                self._chat_history.append('user', prompt, use_cache=True)
                embedding, _ = self._chat_history.embed_chat()
                
                # Build generation kwargs
                gen_kwargs = {
                    'kv_cache': self._chat_history.kv_cache,
                }
                if self.max_new_tokens is not None:
                    gen_kwargs['max_new_tokens'] = self.max_new_tokens
                if self.min_new_tokens is not None:
                    gen_kwargs['min_new_tokens'] = self.min_new_tokens
                if self.do_sample is not None:
                    gen_kwargs['do_sample'] = self.do_sample
                if self.repetition_penalty is not None:
                    gen_kwargs['repetition_penalty'] = self.repetition_penalty
                if self.temperature is not None:
                    gen_kwargs['temperature'] = self.temperature
                if self.top_p is not None:
                    gen_kwargs['top_p'] = self.top_p
                
                reply = self._model.generate(embedding, **gen_kwargs)
                
                # Collect the full reply
                full_reply = ""
                for token in reply:
                    full_reply += token
                
                # Clean up markup from the output
                clean_reply = self._clean_markup(full_reply)
                
                # Push output to stack with prompt
                self._push_output(prompt, clean_reply)
                
                self._chat_history.append('bot', reply)
            
            time_elapsed = time.perf_counter() - time_begin
                        
            self._chat_history.reset()
            
            # Check for end of stream
            if self._video_source.eos:
                break
        
        self._run_flag = False
    
    def shutDown(self):
        """Signal the main processing loop to stop."""
        self._run_flag = False
        self._enabled_event.set()  # Wake up if waiting
    
    def is_running(self):
        """Check if the processing loop is currently running."""
        return self._run_flag
    
    def set_prompts(self, prompts):
        """
        Update the prompts to be applied to each frame.
        Also enables processing if currently disabled.
        
        Args:
            prompts: List of prompt strings
        """
        with self._lock:
            self._prompts = list(prompts)
            self._last_get_output_time = time.time()
            if not self._enabled:
                self._enabled = True
        self._enabled_event.set()  # Wake up the main loop
    
    def get_prompts(self):
        """
        Get the current list of prompts.
        
        Returns:
            List of prompt strings
        """
        with self._lock:
            return list(self._prompts)
    
    def _clean_markup(self, text):
        """Remove </s> end-of-sequence marker from output."""
        if text.endswith('</s>'):
            return text[:-4].strip()
        return text.strip()
    
    def _push_output(self, prompt, output):
        """
        Push an output to the stack as a dict with prompt and output.
        
        Skips pushing if prompts are empty (stale output from before disable).
        Also tracks sameAsNext logic for consecutive identical outputs.
        
        Args:
            prompt: The prompt that generated this output
            output: Output string from the model
        """
        with self._lock:
            # Don't push if prompts are empty - this output is stale
            # (from before disable/auto-disable cleared prompts)
            if not self._prompts:
                return
            
            # Also verify this prompt is still in the current prompts list
            if prompt not in self._prompts:
                return
            
            entry = {'prompt': prompt, 'output': output}
            
            if self._output_stack:
                # Get the current top entry
                top = self._output_stack[0]
                top_output = top.get('output', '')
                
                # If new output matches the top output, mark top as same
                if output == top_output and not top.get('sameAsNext', False):
                    self._output_stack[0]['sameAsNext'] = True
            
            # Push new entry to front (index 0)
            self._output_stack.appendleft(entry)
    
    def get_output(self, index=0, timeout=5.0, prompt=None):
        """
        Get output at the specified index, optionally waiting for a specific prompt.
        
        Enables processing if disabled, and waits for output to be available
        before returning. Also resets the auto-disable timer.
        
        Args:
            index: Stack index (0 = most recent, default: 0)
            timeout: Max seconds to wait for output (default: 30.0)
            prompt: If provided, wait until the entry at index has this prompt
            
        Returns:
            Dict with keys: 'prompt', 'output', and optionally 'sameAsNext'
            Returns {'prompt': None, 'output': None} if timeout or not found
        """
        # Enable processing if not already enabled
        with self._lock:
            self._last_get_output_time = time.time()
            if not self._enabled:
                self._enabled = True
        self._enabled_event.set()
        
        # Wait for output to be available (with optional prompt matching)
        start_time = time.time()
        while True:
            with self._lock:
                self._last_get_output_time = time.time()
                if len(self._output_stack) > index:
                    entry = self._output_stack[index]
                    # If no prompt filter, or prompt matches, return
                    if prompt is None or entry.get('prompt') == prompt:
                        return dict(entry)  # Return a copy
            
            # Check timeout
            if time.time() - start_time >= timeout:
                return {'prompt': None, 'output': None}
            
            # Wait a short time before checking again
            time.sleep(0.1)
    
    def get_output_stack_size(self):
        """Get the current number of items in the output stack."""
        with self._lock:
            return len(self._output_stack)
    
    def clear_output_stack(self):
        """Clear all outputs from the stack."""
        with self._lock:
            self._output_stack.clear()

def main():
    """Main function for standalone execution."""
    # Parse args and set some defaults
    args = ArgParser(extras=ArgParser.Defaults + ['prompt', 'video_input']).parse_args()
    prompts = load_prompts(args.prompt)
    
    if not prompts:
        prompts = ["Describe the image.", "Are there people in the image?"]
    
    if not args.model:
        args.model = "Efficient-Large-Model/VILA1.5-3b"
    
    if not args.video_input:
        args.video_input = "/dev/video0" 
    
    print(f"Configuration: {args}")
    
    # Create NanoVlm instance
    vlm = NanoVlm(
        model=args.model,
        api=args.api,
        quantization=args.quantization,
        max_context_len=args.max_context_len,
        vision_model=args.vision_model,
        vision_scaling=args.vision_scaling,
        chat_template=args.chat_template,
        system_prompt=args.system_prompt,
        video_input=args.video_input,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        prompts=prompts,
    )
    
    # Enable processing for standalone mode
    vlm.enable()
    
    # Run directly (not threaded for standalone mode)
    print("Starting NanoVlm...")
    try:
        vlm.startUp()
    except KeyboardInterrupt:
        print("\nShutting down...")
        vlm.shutDown()
    
    # Print final outputs
    print("\nFinal output stack:")
    for i in range(vlm.get_output_stack_size()):
        result = vlm.get_output(i)
        print(f"  [{i}] prompt: {result.get('prompt')}")
        print(f"       output: {result.get('output')}")
        if result.get('sameAsNext'):
            print(f"       (sameAsNext: True)")

if __name__ == "__main__":
    main()
