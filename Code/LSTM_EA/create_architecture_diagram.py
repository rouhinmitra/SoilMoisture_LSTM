#!/usr/bin/env python
"""
Create architecture diagram comparing LSTM vs EA-LSTM.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

def create_lstm_diagram(ax, title="Standard LSTM"):
    """Create standard LSTM diagram."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Colors
    input_color = '#3498db'  # Blue
    static_color = '#e74c3c'  # Red
    gate_color = '#f39c12'   # Orange
    cell_color = '#2ecc71'   # Green
    output_color = '#9b59b6' # Purple
    
    # Input boxes
    # Dynamic features
    dynamic_box = FancyBboxPatch((0.5, 0.5), 2.5, 1.2, boxstyle="round,pad=0.1",
                                  facecolor=input_color, edgecolor='black', linewidth=2)
    ax.add_patch(dynamic_box)
    ax.text(1.75, 1.1, 'Dynamic\nFeatures\n(SSM, Weather)', ha='center', va='center', 
            fontsize=8, fontweight='bold', color='white')
    
    # Static features
    static_box = FancyBboxPatch((3.5, 0.5), 2.5, 1.2, boxstyle="round,pad=0.1",
                                 facecolor=static_color, edgecolor='black', linewidth=2)
    ax.add_patch(static_box)
    ax.text(4.75, 1.1, 'Static\nFeatures\n(AE Embeddings)', ha='center', va='center', 
            fontsize=8, fontweight='bold', color='white')
    
    # Concatenation
    concat_box = FancyBboxPatch((2, 2.5), 2.5, 0.8, boxstyle="round,pad=0.1",
                                 facecolor='#95a5a6', edgecolor='black', linewidth=2)
    ax.add_patch(concat_box)
    ax.text(3.25, 2.9, 'Concatenate', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    # Arrows to concat
    ax.annotate('', xy=(2.5, 2.5), xytext=(1.75, 1.7),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(4, 2.5), xytext=(4.75, 1.7),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # LSTM Cell
    cell_box = FancyBboxPatch((1.5, 4), 4, 3, boxstyle="round,pad=0.1",
                               facecolor='#ecf0f1', edgecolor='black', linewidth=3)
    ax.add_patch(cell_box)
    ax.text(3.5, 6.7, 'LSTM Cell', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # Gates inside cell
    gate_width = 0.8
    gate_height = 0.6
    
    # Input gate
    ig = FancyBboxPatch((1.8, 5.2), gate_width, gate_height, boxstyle="round,pad=0.05",
                         facecolor=gate_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(ig)
    ax.text(2.2, 5.5, 'iₜ', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Forget gate
    fg = FancyBboxPatch((2.85, 5.2), gate_width, gate_height, boxstyle="round,pad=0.05",
                         facecolor=gate_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(fg)
    ax.text(3.25, 5.5, 'fₜ', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Output gate
    og = FancyBboxPatch((3.9, 5.2), gate_width, gate_height, boxstyle="round,pad=0.05",
                         facecolor=gate_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(og)
    ax.text(4.3, 5.5, 'oₜ', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Cell state
    cs = FancyBboxPatch((2.5, 4.3), 1.5, 0.6, boxstyle="round,pad=0.05",
                         facecolor=cell_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(cs)
    ax.text(3.25, 4.6, 'cₜ', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Arrow from concat to cell
    ax.annotate('', xy=(3.5, 4), xytext=(3.25, 3.3),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Hidden state output
    hidden_box = FancyBboxPatch((2.25, 8), 2, 0.8, boxstyle="round,pad=0.1",
                                 facecolor=output_color, edgecolor='black', linewidth=2)
    ax.add_patch(hidden_box)
    ax.text(3.25, 8.4, 'hₜ (hidden)', ha='center', va='center', 
            fontsize=9, fontweight='bold', color='white')
    
    # Arrow from cell to hidden
    ax.annotate('', xy=(3.25, 8), xytext=(3.5, 7),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Output
    output_box = FancyBboxPatch((2.25, 9.5), 2, 0.8, boxstyle="round,pad=0.1",
                                 facecolor='#1abc9c', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(3.25, 9.9, 'RZSM\nPrediction', ha='center', va='center', 
            fontsize=9, fontweight='bold', color='white')
    
    # Arrow from hidden to output
    ax.annotate('', xy=(3.25, 9.5), xytext=(3.25, 8.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Key difference annotation
    ax.text(7, 3, 'Static features\nare just another\ninput dimension', 
            ha='center', va='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def create_ealstm_diagram(ax, title="Entity-Aware LSTM (EA-LSTM)"):
    """Create EA-LSTM diagram."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Colors
    input_color = '#3498db'  # Blue
    static_color = '#e74c3c'  # Red
    gate_color = '#f39c12'   # Orange
    cell_color = '#2ecc71'   # Green
    output_color = '#9b59b6' # Purple
    special_color = '#e74c3c'  # Red for entity gate
    
    # Dynamic features
    dynamic_box = FancyBboxPatch((0.5, 0.5), 2.5, 1.2, boxstyle="round,pad=0.1",
                                  facecolor=input_color, edgecolor='black', linewidth=2)
    ax.add_patch(dynamic_box)
    ax.text(1.75, 1.1, 'Dynamic\nFeatures\n(SSM, Weather)', ha='center', va='center', 
            fontsize=8, fontweight='bold', color='white')
    
    # Static features (positioned differently)
    static_box = FancyBboxPatch((6, 3.5), 2.5, 1.2, boxstyle="round,pad=0.1",
                                 facecolor=static_color, edgecolor='black', linewidth=2)
    ax.add_patch(static_box)
    ax.text(7.25, 4.1, 'Static\nFeatures\n(AE Embeddings)', ha='center', va='center', 
            fontsize=8, fontweight='bold', color='white')
    
    # Arrow from dynamic to cell
    ax.annotate('', xy=(2.5, 4), xytext=(1.75, 1.7),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # LSTM Cell
    cell_box = FancyBboxPatch((1, 4), 5, 3, boxstyle="round,pad=0.1",
                               facecolor='#ecf0f1', edgecolor='black', linewidth=3)
    ax.add_patch(cell_box)
    ax.text(3.5, 6.7, 'EA-LSTM Cell', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # Gates inside cell
    gate_width = 0.9
    gate_height = 0.7
    
    # Input modulation gate (special - controlled by static)
    img = FancyBboxPatch((1.3, 5.2), gate_width, gate_height, boxstyle="round,pad=0.05",
                          facecolor=special_color, edgecolor='black', linewidth=2)
    ax.add_patch(img)
    ax.text(1.75, 5.55, 'îₜ', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Input gate
    ig = FancyBboxPatch((2.4, 5.2), gate_width, gate_height, boxstyle="round,pad=0.05",
                         facecolor=gate_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(ig)
    ax.text(2.85, 5.55, 'iₜ', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Forget gate
    fg = FancyBboxPatch((3.5, 5.2), gate_width, gate_height, boxstyle="round,pad=0.05",
                         facecolor=gate_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(fg)
    ax.text(3.95, 5.55, 'fₜ', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Output gate
    og = FancyBboxPatch((4.6, 5.2), gate_width, gate_height, boxstyle="round,pad=0.05",
                         facecolor=gate_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(og)
    ax.text(5.05, 5.55, 'oₜ', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Cell state
    cs = FancyBboxPatch((2.75, 4.3), 1.5, 0.6, boxstyle="round,pad=0.05",
                         facecolor=cell_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(cs)
    ax.text(3.5, 4.6, 'cₜ', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Arrow from static to input modulation gate (the key difference!)
    ax.annotate('', xy=(2.2, 5.55), xytext=(6, 4.1),
                arrowprops=dict(arrowstyle='->', color=special_color, lw=3,
                               connectionstyle="arc3,rad=-0.2"))
    
    # Label for the special connection
    ax.text(4.5, 4.9, 'Controls\nInput Gate!', ha='center', va='center', 
            fontsize=8, fontweight='bold', color=special_color,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=special_color, alpha=0.9))
    
    # Hidden state output
    hidden_box = FancyBboxPatch((2.5, 8), 2, 0.8, boxstyle="round,pad=0.1",
                                 facecolor=output_color, edgecolor='black', linewidth=2)
    ax.add_patch(hidden_box)
    ax.text(3.5, 8.4, 'hₜ (hidden)', ha='center', va='center', 
            fontsize=9, fontweight='bold', color='white')
    
    # Arrow from cell to hidden
    ax.annotate('', xy=(3.5, 8), xytext=(3.5, 7),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Output
    output_box = FancyBboxPatch((2.5, 9.5), 2, 0.8, boxstyle="round,pad=0.1",
                                 facecolor='#1abc9c', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(3.5, 9.9, 'RZSM\nPrediction', ha='center', va='center', 
            fontsize=9, fontweight='bold', color='white')
    
    # Arrow from hidden to output
    ax.annotate('', xy=(3.5, 9.5), xytext=(3.5, 8.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Key difference annotation
    ax.text(7.5, 6.5, 'Static features\nMODULATE how\ndynamic features\nenter the cell!', 
            ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#fadbd8', edgecolor=special_color, alpha=0.9))


def create_comparison_figure():
    """Create side-by-side comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    create_lstm_diagram(axes[0])
    create_ealstm_diagram(axes[1])
    
    # Add legend at bottom
    legend_elements = [
        mpatches.Patch(facecolor='#3498db', edgecolor='black', label='Dynamic Features (time-varying)'),
        mpatches.Patch(facecolor='#e74c3c', edgecolor='black', label='Static Features (site characteristics)'),
        mpatches.Patch(facecolor='#f39c12', edgecolor='black', label='Standard Gates (i, f, o)'),
        mpatches.Patch(facecolor='#2ecc71', edgecolor='black', label='Cell State'),
        mpatches.Patch(facecolor='#9b59b6', edgecolor='black', label='Hidden State'),
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, 0.02))
    
    # Main title
    fig.suptitle('LSTM vs Entity-Aware LSTM (EA-LSTM) Architecture Comparison', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Subtitle explaining key difference
    fig.text(0.5, 0.93, 
             'Key Difference: In EA-LSTM, static features (site characteristics) control the input gate,\n'
             'allowing the model to learn site-specific responses to dynamic inputs.',
             ha='center', fontsize=11, style='italic')
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.90])
    
    # Save
    plt.savefig('outputs/lstm_vs_ealstm_architecture.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('outputs/lstm_vs_ealstm_architecture.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print("Architecture diagram saved to:")
    print("  - outputs/lstm_vs_ealstm_architecture.png")
    print("  - outputs/lstm_vs_ealstm_architecture.pdf")
    
    plt.show()


if __name__ == "__main__":
    import os
    os.makedirs('outputs', exist_ok=True)
    create_comparison_figure()

