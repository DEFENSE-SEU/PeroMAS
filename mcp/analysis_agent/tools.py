"""
Analysis Agent Tools - Crystal Structure Visualization

Uses Plotly for interactive 3D visualization.
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from ase.io import read

try:
    from pymatgen.io.cif import CifParser
    from pymatgen.io.ase import AseAtomsAdaptor
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False

try:
    from .visualization_plotly import PlotlyCrystalVisualizer
except ImportError:
    from visualization_plotly import PlotlyCrystalVisualizer


class AnalysisTools:
    """Crystal structure analysis and visualization tools."""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("analysis_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz = PlotlyCrystalVisualizer()
    
    def visualize(
        self,
        cif_content: str,
        name: Optional[str] = None,
        supercell: Optional[tuple] = None,
        theme: str = 'light',
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        Visualize crystal structure as interactive HTML.
        
        Args:
            cif_content: CIF file content
            name: Structure name
            supercell: Supercell dimensions (nx, ny, nz)
            theme: 'light' or 'dark'
            save: Save to file
            
        Returns:
            Dict with filepath and info
        """
        result = {"success": False, "filepath": None, "info": None, "error": None}
        
        try:
            info = self._get_info(cif_content)
            result["info"] = info
            
            if name is None:
                name = info['formula'].replace(" ", "_")
            
            self.viz = PlotlyCrystalVisualizer(theme=theme)
            fig = self.viz.visualize(cif_content, supercell=supercell, title=f"{name}")
            
            if save:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = self.output_dir / f"{name}_{ts}.html"
                self.viz.save_html(fig, filepath)
                result["filepath"] = str(filepath)
            
            result["success"] = True
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def analyze(self, cif_content: str) -> Dict[str, Any]:
        """Get structure information."""
        result = {"success": False, "data": None, "error": None}
        try:
            info = self._get_info(cif_content)
            info["is_perovskite"] = self._check_perovskite(info)
            result["data"] = info
            result["success"] = True
        except Exception as e:
            result["error"] = str(e)
        return result
    
    def _get_info(self, cif_content: str) -> Dict[str, Any]:
        """Parse CIF and extract info."""
        atoms = self._parse_cif(cif_content)
        cell = atoms.get_cell()
        
        return {
            "formula": atoms.get_chemical_formula(),
            "n_atoms": len(atoms),
            "elements": list(set(atoms.get_chemical_symbols())),
            "volume": atoms.get_volume(),
            "cell": {
                "a": cell.lengths()[0],
                "b": cell.lengths()[1], 
                "c": cell.lengths()[2],
            }
        }
    
    def _parse_cif(self, cif_content: str):
        """Parse CIF string to ASE Atoms."""
        if "\\n" in cif_content:
            cif_content = cif_content.replace("\\n", "\n")
        
        if HAS_PYMATGEN:
            try:
                parser = CifParser.from_string(cif_content)
                structure = parser.get_structures()[0]
                return AseAtomsAdaptor.get_atoms(structure)
            except:
                pass
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as f:
            f.write(cif_content)
            temp_path = f.name
        try:
            return read(temp_path, format='cif')
        finally:
            os.unlink(temp_path)
    
    def _check_perovskite(self, info: Dict) -> bool:
        """Check if structure is perovskite-like."""
        elements = set(info['elements'])
        a_site = {'Cs', 'Rb', 'K', 'Na', 'Ba', 'Sr', 'Ca', 'La'}
        b_site = {'Pb', 'Sn', 'Ge', 'Ti', 'Zr', 'Fe', 'Mn'}
        x_site = {'I', 'Br', 'Cl', 'F', 'O'}
        return bool(elements & a_site) and bool(elements & b_site) and bool(elements & x_site)
