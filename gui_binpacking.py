import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import random
import threading
from ga_woc_binpacking import BinPackingGAWoC, generate_random_items
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class BinPackingGUI:
    """
    GUI Application for Bin Packing with GA and GA+WoC Algorithms.
    
    Allows users to switch between standard GA and GA with Wisdom of Crowds (WoC).
    """
    
    def __init__(self, root):
        """Initialize the GUI."""
        self.root = root
        self.root.title("Bin Packing Problem - GA/GA+WoC Solver")
        self.root.geometry("1200x800")
        
        # Variables
        self.items = []
        self.bin_capacity = tk.DoubleVar(value=1.0)
        self.num_items = tk.IntVar(value=30)
        self.min_size = tk.DoubleVar(value=0.1)
        self.max_size = tk.DoubleVar(value=0.8)
        self.population_size = tk.IntVar(value=100)
        self.generations = tk.IntVar(value=200)
        self.mutation_rate = tk.DoubleVar(value=0.1)
        self.crossover_rate = tk.DoubleVar(value=0.8)
        self.crowd_size = tk.IntVar(value=5)
        self.use_woc = tk.BooleanVar(value=True)  # Enable WoC by default
        
        self.solution_bins = []
        self.fitness_history = []
        self.is_running = False
        
        self._create_widgets()
        
    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Left panel - Configuration
        self._create_config_panel(main_frame)
        
        # Right panel - Visualization and Results
        self._create_visualization_panel(main_frame)
        
    def _create_config_panel(self, parent):
        """Create configuration panel."""
        config_frame = ttk.LabelFrame(parent, text="Configuration", padding="10")
        config_frame.grid(row=0, column=0, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        row = 0
        
        # Problem Parameters
        ttk.Label(config_frame, text="PROBLEM PARAMETERS", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, pady=(0, 10))
        row += 1
        
        ttk.Label(config_frame, text="Bin Capacity:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(config_frame, textvariable=self.bin_capacity, width=15).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        ttk.Label(config_frame, text="Number of Items:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(config_frame, textvariable=self.num_items, width=15).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        ttk.Label(config_frame, text="Min Item Size:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(config_frame, textvariable=self.min_size, width=15).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        ttk.Label(config_frame, text="Max Item Size:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(config_frame, textvariable=self.max_size, width=15).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        # Generate items button
        ttk.Button(config_frame, text="Generate Random Items", command=self.generate_items).grid(
            row=row, column=0, columnspan=2, pady=10)
        row += 1
        
        # Custom items button
        ttk.Button(config_frame, text="Enter Custom Items", command=self.enter_custom_items).grid(
            row=row, column=0, columnspan=2, pady=5)
        row += 1
        
        # Separator
        ttk.Separator(config_frame, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # Algorithm Parameters
        ttk.Label(config_frame, text="ALGORITHM PARAMETERS", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, pady=(0, 10))
        row += 1
        
        ttk.Label(config_frame, text="Population Size:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(config_frame, textvariable=self.population_size, width=15).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        ttk.Label(config_frame, text="Generations:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(config_frame, textvariable=self.generations, width=15).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        ttk.Label(config_frame, text="Mutation Rate:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(config_frame, textvariable=self.mutation_rate, width=15).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        ttk.Label(config_frame, text="Crossover Rate:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(config_frame, textvariable=self.crossover_rate, width=15).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        # WoC Enable/Disable checkbox
        self.woc_checkbox = ttk.Checkbutton(
            config_frame, 
            text="Enable WoC (Wisdom of Crowds)",
            variable=self.use_woc,
            command=self._toggle_woc
        )
        self.woc_checkbox.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1
        
        ttk.Label(config_frame, text="Crowd Size:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.crowd_size_entry = ttk.Entry(config_frame, textvariable=self.crowd_size, width=15)
        self.crowd_size_entry.grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        # Separator
        ttk.Separator(config_frame, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1
        
        # Solve button
        self.solve_button = ttk.Button(config_frame, text="Solve with GA+WoC", 
                                       command=self.solve_problem, style='Accent.TButton')
        self.solve_button.grid(row=row, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        row += 1
        
        # Progress bar (determinate mode for actual progress)
        self.progress = ttk.Progressbar(config_frame, mode='determinate', maximum=100)
        self.progress.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        # Progress label
        self.progress_label = ttk.Label(config_frame, text="Ready", font=('Arial', 8))
        self.progress_label.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        row += 1
        
        # Current items display
        ttk.Label(config_frame, text="Current Items:", font=('Arial', 9, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        row += 1
        
        self.items_text = scrolledtext.ScrolledText(config_frame, width=30, height=8, wrap=tk.WORD)
        self.items_text.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
    
    def _toggle_woc(self):
        """Toggle WoC enable/disable and update UI."""
        if self.use_woc.get():
            self.crowd_size_entry.config(state='normal')
            self.solve_button.config(text="Solve with GA+WoC")
        else:
            self.crowd_size_entry.config(state='disabled')
            self.solve_button.config(text="Solve with GA")
        
    def _create_visualization_panel(self, parent):
        """Create visualization panel."""
        viz_frame = ttk.Frame(parent)
        viz_frame.grid(row=0, column=1, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(1, weight=1)
        
        # Results text
        results_frame = ttk.LabelFrame(viz_frame, text="Solution Results", padding="10")
        results_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, width=80, height=8, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Visualization tabs
        self.notebook = ttk.Notebook(viz_frame)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Bin visualization tab
        self.bin_viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.bin_viz_frame, text="Bin Visualization")
        
        # Fitness history tab
        self.fitness_viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.fitness_viz_frame, text="Fitness History")
        
        # Create initial empty plots
        self._create_bin_visualization()
        self._create_fitness_visualization()
        
    def _create_bin_visualization(self):
        """Create bin visualization plot."""
        self.bin_fig = Figure(figsize=(8, 6))
        self.bin_ax = self.bin_fig.add_subplot(111)
        self.bin_canvas = FigureCanvasTkAgg(self.bin_fig, master=self.bin_viz_frame)
        self.bin_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.bin_ax.set_title("Bin Packing Solution")
        self.bin_ax.text(0.5, 0.5, 'No solution yet.\nClick "Solve with GA+WoC" to start.',
                        ha='center', va='center', transform=self.bin_ax.transAxes, fontsize=12)
        self.bin_ax.set_xticks([])
        self.bin_ax.set_yticks([])
        
    def _create_fitness_visualization(self):
        """Create fitness history plot."""
        self.fitness_fig = Figure(figsize=(8, 6))
        self.fitness_ax = self.fitness_fig.add_subplot(111)
        self.fitness_canvas = FigureCanvasTkAgg(self.fitness_fig, master=self.fitness_viz_frame)
        self.fitness_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.fitness_ax.set_title("Fitness Evolution")
        self.fitness_ax.set_xlabel("Generation")
        self.fitness_ax.set_ylabel("Fitness (Lower is Better)")
        self.fitness_ax.grid(True, alpha=0.3)
        
    def generate_items(self):
        """Generate random items."""
        try:
            num = self.num_items.get()
            min_s = self.min_size.get()
            max_s = self.max_size.get()
            
            if min_s >= max_s:
                messagebox.showerror("Error", "Min size must be less than max size")
                return
            
            self.items = generate_random_items(num, min_s, max_s)
            self._update_items_display()
            messagebox.showinfo("Success", f"Generated {num} random items")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate items: {str(e)}")
    
    def enter_custom_items(self):
        """Open dialog to enter custom items."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Enter Custom Items")
        dialog.geometry("400x300")
        
        ttk.Label(dialog, text="Enter item sizes (one per line):").pack(pady=10)
        
        text_widget = scrolledtext.ScrolledText(dialog, width=40, height=15)
        text_widget.pack(padx=10, pady=10)
        
        if self.items:
            text_widget.insert('1.0', '\n'.join(str(item) for item in self.items))
        
        def save_items():
            try:
                content = text_widget.get('1.0', tk.END)
                items = [float(line.strip()) for line in content.split('\n') if line.strip()]
                
                if not items:
                    messagebox.showerror("Error", "No valid items entered")
                    return
                
                if any(item <= 0 for item in items):
                    messagebox.showerror("Error", "All items must be positive")
                    return
                
                self.items = items
                self._update_items_display()
                dialog.destroy()
                messagebox.showinfo("Success", f"Loaded {len(items)} custom items")
                
            except ValueError:
                messagebox.showerror("Error", "Invalid input. Please enter numeric values only.")
        
        ttk.Button(dialog, text="Save Items", command=save_items).pack(pady=10)
    
    def _update_items_display(self):
        """Update the items text display."""
        self.items_text.delete('1.0', tk.END)
        if self.items:
            items_str = ', '.join(f"{item:.2f}" for item in self.items[:50])
            if len(self.items) > 50:
                items_str += f"\n... and {len(self.items) - 50} more items"
            self.items_text.insert('1.0', f"Total: {len(self.items)} items\nSizes: {items_str}")
    
    def solve_problem(self):
        """Solve the bin packing problem."""
        if self.is_running:
            messagebox.showwarning("Warning", "Solver is already running")
            return
        
        if not self.items:
            messagebox.showerror("Error", "Please generate or enter items first")
            return
        
        # Validate that items fit in bins
        capacity = self.bin_capacity.get()
        if any(item > capacity for item in self.items):
            messagebox.showerror("Error", f"Some items are larger than bin capacity ({capacity})")
            return
        
        # Run solver in separate thread
        self.is_running = True
        self.solve_button.config(state='disabled')
        self.progress['value'] = 0
        self.progress_label.config(text="Initializing...")
        
        thread = threading.Thread(target=self._run_solver)
        thread.daemon = True
        thread.start()
    
    def _update_progress(self, current, total):
        """Update progress bar (called from solver thread)."""
        progress_percent = (current / total) * 100
        self.root.after(0, self._set_progress, progress_percent, current, total)
    
    def _set_progress(self, percent, current, total):
        """Set progress bar value (must be called from main thread)."""
        self.progress['value'] = percent
        self.progress_label.config(text=f"Generation {current}/{total} ({percent:.1f}%)")
    
    def _run_solver(self):
        """Run the GA+WoC solver (in separate thread)."""
        try:
            solver = BinPackingGAWoC(
                items=self.items,
                bin_capacity=self.bin_capacity.get(),
                population_size=self.population_size.get(),
                generations=self.generations.get(),
                mutation_rate=self.mutation_rate.get(),
                crossover_rate=self.crossover_rate.get(),
                crowd_size=self.crowd_size.get(),
                use_woc=self.use_woc.get()
            )
            
            bins, num_bins, fitness_history, computation_time = solver.solve(
                verbose=False, 
                progress_callback=self._update_progress
            )
            
            # Update GUI in main thread
            self.root.after(0, self._display_solution, bins, num_bins, fitness_history, computation_time)
            
        except Exception as e:
            self.root.after(0, self._handle_solver_error, str(e))
    
    def _display_solution(self, bins, num_bins, fitness_history, computation_time):
        """Display the solution in GUI."""
        self.solution_bins = bins
        self.fitness_history = fitness_history
        
        algo_name = "GA+WoC" if self.use_woc.get() else "GA"
        
        # Calculate average utilization
        total_utilization = 0
        for bin_items in bins:
            load = sum(bin_items)
            utilization = (load / self.bin_capacity.get()) * 100
            total_utilization += utilization
        avg_utilization = total_utilization / num_bins if num_bins > 0 else 0
        
        # Update results text
        self.results_text.delete('1.0', tk.END)
        results = f"=== SOLUTION FOUND ({algo_name}) ===\n\n"
        results += f"Algorithm: {algo_name}\n"
        results += f"Number of bins used: {num_bins}\n"
        results += f"Final fitness: {fitness_history[-1]:.4f}\n"
        results += f"Computation time: {computation_time:.2f} seconds\n"
        results += f"Total items: {len(self.items)}\n"
        results += f"Bin capacity: {self.bin_capacity.get()}\n"
        results += f"Average utilization: {avg_utilization:.2f}%\n"
        if self.use_woc.get():
            results += f"Crowd size: {self.crowd_size.get()}\n"
        results += "\n"
        
        results += "Bin contents:\n"
        for i, bin_items in enumerate(bins):
            load = sum(bin_items)
            utilization = (load / self.bin_capacity.get()) * 100
            results += f"Bin {i+1}: {len(bin_items)} items, Load: {load:.2f}/{self.bin_capacity.get()} ({utilization:.1f}%)\n"
            results += f"  Items: {', '.join(f'{item:.2f}' for item in bin_items)}\n"
        
        self.results_text.insert('1.0', results)
        
        # Update visualizations
        self._plot_bins(bins)
        self._plot_fitness(fitness_history)
        
        # Re-enable button and reset progress
        self.is_running = False
        self.solve_button.config(state='normal')
        self.progress['value'] = 100
        self.progress_label.config(text="Completed!")
        
        messagebox.showinfo("Success", f"Solution found using {num_bins} bins with {algo_name}!")
    
    def _plot_bins(self, bins):
        """Plot the bin packing solution."""
        self.bin_ax.clear()
        
        if not bins:
            return
        
        num_bins = len(bins)
        capacity = self.bin_capacity.get()
        colors = plt.cm.Set3(range(20))
        
        # Create bar chart
        bin_numbers = list(range(1, num_bins + 1))
        
        # Stack items in each bin
        bottom = [0] * num_bins
        
        # Group items by size for better visualization
        max_items_per_bin = max(len(b) for b in bins)
        
        for item_idx in range(max_items_per_bin):
            heights = []
            for bin_items in bins:
                if item_idx < len(bin_items):
                    heights.append(bin_items[item_idx])
                else:
                    heights.append(0)
            
            self.bin_ax.bar(bin_numbers, heights, bottom=bottom, 
                          color=colors[item_idx % len(colors)],
                          edgecolor='black', linewidth=0.5)
            
            bottom = [b + h for b, h in zip(bottom, heights)]
        
        # Add capacity line
        self.bin_ax.axhline(y=capacity, color='red', linestyle='--', 
                           linewidth=2, label=f'Capacity Limit: {capacity}')
        
        # Add utilization percentage on top of each bar
        for i, (bin_num, total_load) in enumerate(zip(bin_numbers, bottom)):
            utilization = (total_load / capacity) * 100
            self.bin_ax.text(bin_num, total_load + capacity * 0.02, 
                           f'{utilization:.0f}%',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        self.bin_ax.set_xlabel('Bin Number', fontsize=10, fontweight='bold')
        self.bin_ax.set_ylabel('Load', fontsize=10, fontweight='bold')
        self.bin_ax.set_title(f'Bin Packing Visualization - {num_bins} bins used', 
                            fontsize=11, fontweight='bold')
        self.bin_ax.set_xticks(bin_numbers)
        
        # Set y-axis limit to show capacity clearly
        self.bin_ax.set_ylim(0, capacity * 1.15)
        
        self.bin_ax.legend(loc='upper right')
        self.bin_ax.grid(True, alpha=0.3, axis='y')
        
        # Add explanation text
        self.bin_ax.text(0.02, 0.98, 
                        'Note: Each bar represents a bin, different colors show different items.\nPercentage above each bar shows utilization rate.',
                        transform=self.bin_ax.transAxes,
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.bin_fig.tight_layout()
        self.bin_canvas.draw()
    
    def _plot_fitness(self, fitness_history):
        """Plot fitness evolution."""
        self.fitness_ax.clear()
        
        if not fitness_history:
            return
        
        generations = list(range(len(fitness_history)))
        
        self.fitness_ax.plot(generations, fitness_history, 'b-', linewidth=2, label='Best Fitness')
        self.fitness_ax.set_xlabel('Generation', fontsize=10)
        self.fitness_ax.set_ylabel('Fitness (Number of Bins)', fontsize=10)
        self.fitness_ax.set_title('Fitness Evolution Over Generations', fontsize=12, fontweight='bold')
        self.fitness_ax.grid(True, alpha=0.3)
        self.fitness_ax.legend()
        
        # Fix y-axis formatting to avoid scientific notation
        self.fitness_ax.ticklabel_format(style='plain', axis='y', useOffset=False)
        
        # Add final value annotation
        final_fitness = fitness_history[-1]
        self.fitness_ax.annotate(f'Final: {final_fitness:.2f}',
                               xy=(len(fitness_history)-1, final_fitness),
                               xytext=(-50, 20), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        self.fitness_fig.tight_layout()
        self.fitness_canvas.draw()
    
    def _handle_solver_error(self, error_msg):
        """Handle errors from solver."""
        self.is_running = False
        self.solve_button.config(state='normal')
        self.progress['value'] = 0
        self.progress_label.config(text="Error occurred")
        messagebox.showerror("Solver Error", f"An error occurred:\n{error_msg}")


def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    app = BinPackingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
