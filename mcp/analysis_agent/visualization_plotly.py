"""
Crystal Structure Visualization - Plotly Edition

Interactive 3D crystal structure visualization using Plotly
- Beautiful glass-like spheres with gradients
- Smooth bonds with gradient coloring
- Interactive rotation, zoom, pan
- Export to HTML for sharing

Author: PSC_Agents Team
"""

import os
import tempfile
import json
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict, Any

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from ase import Atoms
from ase.io import read

try:
    from pymatgen.io.cif import CifParser
    from pymatgen.io.ase import AseAtomsAdaptor
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False


# Beautiful color scheme for perovskites
ELEMENT_COLORS = {
    # A-site - vibrant colors
    'Cs': 'rgb(142, 68, 173)',   # Purple
    'Rb': 'rgb(155, 89, 182)',   # Light purple
    'K':  'rgb(165, 105, 189)',  # Lavender
    'Na': 'rgb(187, 143, 206)',  # Pink
    'Ba': 'rgb(39, 174, 96)',    # Green
    'Sr': 'rgb(46, 204, 113)',   # Light green
    'Ca': 'rgb(88, 214, 141)',   # Mint
    'La': 'rgb(52, 152, 219)',   # Blue
    
    # B-site - metallic
    'Pb': 'rgb(52, 73, 94)',     # Dark blue-gray
    'Sn': 'rgb(127, 140, 141)',  # Gray
    'Ge': 'rgb(149, 165, 166)',  # Light gray
    'Ti': 'rgb(189, 195, 199)',  # Silver
    'Bi': 'rgb(142, 68, 173)',   # Purple
    
    # X-site - halides (bright)
    'I':  'rgb(231, 76, 60)',    # Red
    'Br': 'rgb(230, 126, 34)',   # Orange  
    'Cl': 'rgb(241, 196, 15)',   # Yellow
    'F':  'rgb(46, 204, 113)',   # Green
    'O':  'rgb(231, 76, 60)',    # Red
    'S':  'rgb(243, 156, 18)',   # Gold
    
    # Other elements
    'C':  'rgb(26, 188, 156)',   # Teal
    'N':  'rgb(52, 152, 219)',   # Blue
    'H':  'rgb(236, 240, 241)',  # White
    'Zn': 'rgb(125, 128, 176)',  # Blue-gray
    'Cu': 'rgb(200, 128, 51)',   # Copper
    'Fe': 'rgb(224, 102, 51)',   # Iron orange
    'Ag': 'rgb(192, 192, 192)',  # Silver
    'Au': 'rgb(255, 215, 0)',    # Gold
}

# Visualization radii
ELEMENT_RADII = {
    'H': 0.35, 'C': 0.70, 'N': 0.65, 'O': 0.60, 'F': 0.50,
    'Na': 1.80, 'K': 2.20, 'Ca': 1.80, 'Ti': 1.40,
    'Cu': 1.35, 'Zn': 1.35, 'Br': 1.15, 'Rb': 2.35,
    'Sr': 2.00, 'Ag': 1.60, 'Sn': 1.45, 'I': 1.40,
    'Cs': 2.60, 'Ba': 2.15, 'La': 1.95, 'Pb': 1.80,
    'Bi': 1.60, 'Ge': 1.25, 'Cl': 1.00, 'S': 1.00,
}


class PlotlyCrystalVisualizer:
    """
    Beautiful interactive crystal visualization with Plotly.
    """
    
    def __init__(self, theme: str = 'light'):
        """
        Initialize visualizer.
        
        Args:
            theme: 'light' or 'dark'
        """
        if not HAS_PLOTLY:
            raise ImportError("Plotly required. Install with: pip install plotly")
        
        self.theme = theme
        self.colors = ELEMENT_COLORS
        self.radii = ELEMENT_RADII
        
        # Theme settings
        if theme == 'dark':
            self.bg_color = 'rgb(17, 17, 17)'
            self.grid_color = 'rgb(50, 50, 50)'
            self.text_color = 'rgb(200, 200, 200)'
            self.cell_color = 'rgba(255, 255, 255, 0.3)'
        else:
            self.bg_color = 'rgb(255, 255, 255)'
            self.grid_color = 'rgb(220, 220, 220)'
            self.text_color = 'rgb(50, 50, 50)'
            self.cell_color = 'rgba(0, 0, 0, 0.3)'
    
    def _parse_cif(self, cif_input: Union[str, Path]) -> Atoms:
        """Parse CIF to ASE Atoms."""
        if isinstance(cif_input, Path) or (isinstance(cif_input, str) and os.path.exists(cif_input)):
            if HAS_PYMATGEN:
                try:
                    parser = CifParser(str(cif_input))
                    structure = parser.get_structures()[0]
                    return AseAtomsAdaptor.get_atoms(structure)
                except:
                    pass
            return read(str(cif_input), format='cif')
        else:
            if "\\n" in cif_input:
                cif_input = cif_input.replace("\\n", "\n")
            
            if HAS_PYMATGEN:
                try:
                    parser = CifParser.from_string(cif_input)
                    structure = parser.get_structures()[0]
                    return AseAtomsAdaptor.get_atoms(structure)
                except:
                    pass
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as f:
                f.write(cif_input)
                temp_path = f.name
            try:
                return read(temp_path, format='cif')
            finally:
                os.unlink(temp_path)
    
    def _get_color(self, symbol: str) -> str:
        """Get element color."""
        return self.colors.get(symbol, 'rgb(128, 128, 128)')
    
    def _get_radius(self, symbol: str) -> float:
        """Get element radius."""
        return self.radii.get(symbol, 1.0)
    
    def _make_sphere(
        self, 
        center: np.ndarray, 
        radius: float, 
        color: str,
        resolution: int = 20
    ) -> go.Mesh3d:
        """Create a sphere mesh."""
        phi = np.linspace(0, np.pi, resolution)
        theta = np.linspace(0, 2 * np.pi, resolution)
        
        phi, theta = np.meshgrid(phi, theta)
        phi = phi.flatten()
        theta = theta.flatten()
        
        x = center[0] + radius * np.sin(phi) * np.cos(theta)
        y = center[1] + radius * np.sin(phi) * np.sin(theta)
        z = center[2] + radius * np.cos(phi)
        
        # Create triangular faces
        points_per_row = resolution
        i_list, j_list, k_list = [], [], []
        
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                p1 = i * points_per_row + j
                p2 = i * points_per_row + j + 1
                p3 = (i + 1) * points_per_row + j
                p4 = (i + 1) * points_per_row + j + 1
                
                i_list.extend([p1, p1])
                j_list.extend([p2, p3])
                k_list.extend([p3, p4])
        
        return go.Mesh3d(
            x=x, y=y, z=z,
            i=i_list, j=j_list, k=k_list,
            color=color,
            opacity=0.95,
            flatshading=False,
            lighting=dict(
                ambient=0.4,
                diffuse=0.8,
                specular=0.6,
                roughness=0.2,
                fresnel=0.2
            ),
            lightposition=dict(x=100, y=200, z=100),
            hoverinfo='skip'
        )
    
    def _create_atom_scatter(
        self,
        positions: np.ndarray,
        symbols: List[str],
        scale: float = 0.4
    ) -> go.Scatter3d:
        """Create scatter plot for atoms (faster than mesh spheres)."""
        colors = [self._get_color(s) for s in symbols]
        sizes = [self._get_radius(s) * scale * 30 for s in symbols]  # Scale for marker size
        
        # Create hover text
        hover_text = [f"{s}<br>({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})" 
                     for s, p in zip(symbols, positions)]
        
        return go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.95,
                line=dict(width=1, color='rgba(255,255,255,0.5)'),
                symbol='circle',
            ),
            text=hover_text,
            hoverinfo='text',
            name='Atoms'
        )
    
    def _create_bonds(
        self,
        positions: np.ndarray,
        symbols: List[str],
        cutoff: float = 3.5
    ) -> List[go.Scatter3d]:
        """Create bond lines."""
        bonds = []
        n = len(positions)
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                r1 = self._get_radius(symbols[i])
                r2 = self._get_radius(symbols[j])
                max_bond = (r1 + r2) * 1.3
                
                if dist < min(cutoff, max_bond * 1.5):
                    bonds.append((i, j))
        
        # Create line traces for bonds
        traces = []
        for i, j in bonds:
            p1, p2 = positions[i], positions[j]
            mid = (p1 + p2) / 2
            c1 = self._get_color(symbols[i])
            c2 = self._get_color(symbols[j])
            
            # First half of bond
            traces.append(go.Scatter3d(
                x=[p1[0], mid[0]],
                y=[p1[1], mid[1]],
                z=[p1[2], mid[2]],
                mode='lines',
                line=dict(width=6, color=c1),
                hoverinfo='skip',
                showlegend=False
            ))
            
            # Second half of bond
            traces.append(go.Scatter3d(
                x=[mid[0], p2[0]],
                y=[mid[1], p2[1]],
                z=[mid[2], p2[2]],
                mode='lines',
                line=dict(width=6, color=c2),
                hoverinfo='skip',
                showlegend=False
            ))
        
        return traces
    
    def _create_unit_cell(
        self,
        cell: np.ndarray,
        origin: np.ndarray = None
    ) -> go.Scatter3d:
        """Create unit cell edges."""
        if origin is None:
            origin = np.array([0, 0, 0])
        
        # Unit cell vertices
        v = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ])
        vertices = origin + v @ cell
        
        # Edges (with None to break lines)
        edges = [
            0, 1, 2, 3, 0, None,  # Bottom
            4, 5, 6, 7, 4, None,  # Top
            0, 4, None, 1, 5, None, 2, 6, None, 3, 7  # Vertical
        ]
        
        x, y, z = [], [], []
        for e in edges:
            if e is None:
                x.append(None)
                y.append(None)
                z.append(None)
            else:
                x.append(vertices[e, 0])
                y.append(vertices[e, 1])
                z.append(vertices[e, 2])
        
        return go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(width=3, color=self.cell_color),
            hoverinfo='skip',
            name='Unit Cell'
        )
    
    def visualize(
        self,
        cif_input: Union[str, Path],
        supercell: Optional[Tuple[int, int, int]] = None,
        show_bonds: bool = True,
        show_cell: bool = True,
        atom_scale: float = 0.4,
        title: Optional[str] = None,
        width: int = 900,
        height: int = 700,
    ) -> go.Figure:
        """
        Create interactive 3D visualization.
        
        Args:
            cif_input: CIF file or string
            supercell: Supercell dimensions
            show_bonds: Show bonds
            show_cell: Show unit cell
            atom_scale: Atom size scale
            title: Plot title
            width: Figure width
            height: Figure height
            
        Returns:
            Plotly Figure
        """
        atoms = self._parse_cif(cif_input)
        
        if supercell:
            atoms = atoms.repeat(supercell)
        
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        cell = atoms.get_cell()
        
        # Create traces
        traces = []
        
        # Add atoms
        traces.append(self._create_atom_scatter(positions, symbols, atom_scale))
        
        # Add bonds
        if show_bonds:
            traces.extend(self._create_bonds(positions, symbols))
        
        # Add unit cell
        if show_cell and cell is not None:
            traces.append(self._create_unit_cell(cell, positions.min(axis=0)))
        
        # Create figure
        fig = go.Figure(data=traces)
        
        # Calculate axis ranges
        padding = 1.0
        x_range = [positions[:, 0].min() - padding, positions[:, 0].max() + padding]
        y_range = [positions[:, 1].min() - padding, positions[:, 1].max() + padding]
        z_range = [positions[:, 2].min() - padding, positions[:, 2].max() + padding]
        
        # Make equal aspect ratio
        max_range = max(
            x_range[1] - x_range[0],
            y_range[1] - y_range[0],
            z_range[1] - z_range[0]
        )
        
        mid_x = sum(x_range) / 2
        mid_y = sum(y_range) / 2
        mid_z = sum(z_range) / 2
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title or "Crystal Structure",
                font=dict(size=20, color=self.text_color),
                x=0.5,
                y=0.95
            ),
            scene=dict(
                xaxis=dict(
                    title=dict(text='X (Å)', font=dict(color=self.text_color)),
                    range=[mid_x - max_range/2, mid_x + max_range/2],
                    backgroundcolor=self.bg_color,
                    gridcolor=self.grid_color,
                    showbackground=True,
                    tickfont=dict(color=self.text_color)
                ),
                yaxis=dict(
                    title=dict(text='Y (Å)', font=dict(color=self.text_color)),
                    range=[mid_y - max_range/2, mid_y + max_range/2],
                    backgroundcolor=self.bg_color,
                    gridcolor=self.grid_color,
                    showbackground=True,
                    tickfont=dict(color=self.text_color)
                ),
                zaxis=dict(
                    title=dict(text='Z (Å)', font=dict(color=self.text_color)),
                    range=[mid_z - max_range/2, mid_z + max_range/2],
                    backgroundcolor=self.bg_color,
                    gridcolor=self.grid_color,
                    showbackground=True,
                    tickfont=dict(color=self.text_color)
                ),
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                ),
            ),
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            width=width,
            height=height,
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.3)',
                borderwidth=1
            ),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        
        # Add element legend annotations
        unique_elements = sorted(set(symbols))
        for i, elem in enumerate(unique_elements):
            fig.add_annotation(
                x=0.02,
                y=0.95 - i * 0.05,
                xref='paper',
                yref='paper',
                text=f'● {elem}',
                showarrow=False,
                font=dict(size=14, color=self._get_color(elem)),
                bgcolor='rgba(255,255,255,0.8)',
                borderpad=2
            )
        
        return fig
    
    def save_html(
        self,
        fig: go.Figure,
        filepath: Union[str, Path],
        include_plotlyjs: bool = True
    ) -> Path:
        """Save figure as interactive HTML."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        fig.write_html(
            str(filepath),
            include_plotlyjs='cdn' if not include_plotlyjs else True,
            full_html=True
        )
        
        return filepath
    
    def save_image(
        self,
        fig: go.Figure,
        filepath: Union[str, Path],
        format: str = 'png',
        scale: float = 2.0
    ) -> Path:
        """Save figure as static image."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        fig.write_image(str(filepath), format=format, scale=scale)
        return filepath


def visualize_crystal_interactive(
    cif_input: Union[str, Path],
    output_html: Optional[Union[str, Path]] = None,
    supercell: Optional[Tuple[int, int, int]] = None,
    title: Optional[str] = None,
    theme: str = 'light',
    **kwargs
) -> Union[go.Figure, Path]:
    """
    Quick interactive visualization.
    
    Args:
        cif_input: CIF file or string
        output_html: Save HTML path
        supercell: Supercell
        title: Title
        theme: 'light' or 'dark'
        
    Returns:
        Figure or saved path
    """
    viz = PlotlyCrystalVisualizer(theme=theme)
    fig = viz.visualize(cif_input, supercell=supercell, title=title, **kwargs)
    
    if output_html:
        return viz.save_html(fig, output_html)
    
    return fig


# Testing
if __name__ == "__main__":
    test_cif = """# CsPbI3 cubic perovskite
data_CsPbI3
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   6.28940000
_cell_length_b   6.28940000
_cell_length_c   6.28940000
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   1
_chemical_formula_structural   CsPbI3
_chemical_formula_sum   'Cs1 Pb1 I3'
_cell_volume   248.86851658
_cell_formula_units_Z   1
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Cs  Cs0  1  0.00000000  0.00000000  0.00000000  1
  Pb  Pb1  1  0.50000000  0.50000000  0.50000000  1
  I  I2  1  0.50000000  0.50000000  0.00000000  1
  I  I3  1  0.50000000  0.00000000  0.50000000  1
  I  I4  1  0.00000000  0.50000000  0.50000000  1
"""
    
    print("Testing Plotly Crystal Visualizer...")
    print("=" * 50)
    
    # Light theme
    print("\n1. Creating light theme visualization...")
    viz = PlotlyCrystalVisualizer(theme='light')
    fig = viz.visualize(test_cif, supercell=(2, 2, 2), 
                       title="CsPbI₃ Cubic Perovskite (2×2×2)")
    viz.save_html(fig, "test_plotly_light.html")
    print("   Saved: test_plotly_light.html")
    
    # Dark theme
    print("\n2. Creating dark theme visualization...")
    viz_dark = PlotlyCrystalVisualizer(theme='dark')
    fig_dark = viz_dark.visualize(test_cif, supercell=(2, 2, 2),
                                  title="CsPbI₃ Dark Theme")
    viz_dark.save_html(fig_dark, "test_plotly_dark.html")
    print("   Saved: test_plotly_dark.html")
    
    print("\n✅ Tests completed! Open the HTML files in a browser.")
