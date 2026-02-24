{
        "id": "Q001",
        "type": "phase_stability",
        "query": """We are trying to synthesize all-inorganic CsPbI3, but the black phase is extremely unstable in air. I want to try B-site doping with small divalent metals (like Zn2+ or Mn2+) to stabilize the lattice.

First, search the literature for the most effective B-site dopants for CsPbI3 in the last two years.
Based on this, design a specific CsPb(1-x)MxI3 recipe and outline the annealing protocol.
Predict if the alpha-phase formation energy is lowered enough to be stable at room temperature, and if the PCE exceeds 18%.
Finally, analyze the mechanism: does the dopant stabilize the structure by relaxing lattice strain (Goldschmidt tolerance factor) or by increasing bond strength?"""
    },
    {
        "id": "Q002",
        "type": "phase_stability",
        "query": """To improve thermal stability, we are looking into high-entropy perovskites.

Search for recent multi-cation strategies mixing Cs, MA, FA, Rb, and K.
Design a 'high-entropy' formula with at least 4 cations that maximizes the mixing entropy.
Predict its decomposition temperature and whether the PCE can still reach 22%.
Analyze the thermodynamics: explicitly calculate the T*Delta_S contribution to the Gibbs free energy to prove if the stability is truly entropy-driven."""
    },
    {
        "id": "Q003",
        "type": "phase_stability",
        "query": """For our tandem top cell (1.75 eV), the mixed halide (I/Br) perovskite suffers from phase segregation under illumination.

Search for 'triple cation' (Cs/MA/FA) strategies specifically reported to suppress this segregation.
Design a wide-bandgap recipe utilizing these cations.
Predict the bandgap stability under continuous 1-sun illumination and the steady-state Voc.
Analyze the root cause: use your tools to determine if the suppression comes from immobilizing halide ions or modifying the crystal lattice stiffness."""
    },
    
    # === Group 2: Passivation & Defects (Q004-Q006) ===
    {
        "id": "Q004",
        "type": "passivation",
        "query": """Humidity stability is our main bottleneck. I want to build a 2D/3D heterojunction using hydrophobic large organic cations.

Search for fluorinated organic ammonium salts (like F-PEA) used for surface passivation recently.
Design a process to spin-coat this 2D layer on top of a FAPbI3 bulk film.
Predict the T80 lifetime under 60% relative humidity and any change in Series Resistance (Rs).
Analyze the interface electronics: does the 2D layer create a transport barrier for holes, or does it effectively block moisture ingress?"""
    },
    {
        "id": "Q005",
        "type": "passivation",
        "query": """We suspect the interface between the SnO2 ETL and the perovskite has high defect density. I need a molecular bridge to modify this buried interface.

Search for molecules with bifunctional groups (e.g., carboxyl + amine) that bind to both SnO2 and Pb.
Design a modification protocol for the SnO2 substrate before perovskite deposition.
Predict the improvement in Open-Circuit Voltage (Voc) and Fill Factor.
Analyze the chemical mechanism: confirm if the molecule creates a dipole that shifts the work function to better align the energy levels."""
    },
    {
        "id": "Q006",
        "type": "passivation",
        "query": """Our devices show severe hysteresis due to ion migration. I'm interested in polymer additives that can crosslink grain boundaries.

Search for polymers containing carbonyl (C=O) or nitrile (C≡N) groups used as additives.
Design a precursor ink recipe adding a small amount of the best polymer candidate.
Predict the reduction in the Hysteresis Index and the impact on grain size.
Analyze the interaction: quantitatively estimate the binding energy between the polymer functional groups and the undercoordinated Pb2+ defects."""
    },
    
    # === Group 3: Eco-friendly & Novel Materials (Q007-Q009) ===
    {
        "id": "Q007",
        "type": "eco_friendly",
        "query": """We are developing a bottom cell with 1.25 eV bandgap using Sn-Pb perovskite, but Sn oxidation is killing the efficiency.

Search for novel 'scavenger' additives (like metallic Sn powder or specific reductants) used in high-efficiency Sn-Pb cells.
Design a mixed Sn-Pb recipe incorporating the most promising antioxidant strategy.
Predict the Jsc and the stability in ambient air (T50).
Analyze the mechanism: does the additive actively reduce Sn4+ back to Sn2+, or does it form a protective shell around the grains?"""
    },
    {
        "id": "Q008",
        "type": "eco_friendly",
        "query": """Cs2AgBiBr6 is stable but has a wide indirect bandgap. We need to reduce the bandgap for photovoltaic applications.

Search for doping strategies (e.g., using Sb, Fe, or Tl) specifically aimed at bandgap narrowing for this material.
Design a specific stoichiometry for a doped double perovskite.
Predict the new optical bandgap and whether the transition becomes Direct.
Analyze the electronic structure: which orbitals from the dopant are introducing intermediate states inside the bandgap?"""
    },
    {
        "id": "Q009",
        "type": "eco_friendly",
        "query": """Toxicity of DMF/DMSO is a major issue for industrialization. We need a green solvent system for processing MAPbI3.

Search for solvent systems based on TEP (Triethyl phosphate) or other non-toxic alternatives.
Design a fully green ink formulation and the corresponding quenching method.
Predict the film morphology (roughness) and the final PCE compared to the DMF control.
Analyze the colloid chemistry: compare the coordination number (Gutmann Donor Number) of your green solvent vs DMSO to explain the crystallization kinetics."""
    },
    
    # === Group 4: Processing & Device Architecture (Q010-Q012) ===
    {
        "id": "Q010",
        "type": "processing",
        "query": """We are moving from spin coating to doctor blading for large-area modules. The ink rheology needs to be adjusted.

Search for additives (like surfactants or viscous polymers) used to optimize ink flow for blade coating.
Design a specific ink recipe for a >50 cm² module.
Predict the PCE loss factor when scaling from 0.1 cm² to 50 cm².
Analyze the drying dynamics: explain how the additive suppresses the 'coffee-ring effect' during the slow drying process."""
    },
    {
        "id": "Q011",
        "type": "processing",
        "query": """To cut costs, we want to make HTL-free devices using carbon electrodes. The solvent in the carbon paste often destroys the perovskite.

Search for recent 'solvent-proof' perovskite compositions or protective interlayers for carbon-based PSCs.
Design a robust device stack (FTO/ETL/Perovskite/Carbon).
Predict the stability (T80) and the efficiency potential.
Analyze the charge extraction: calculate the energy barrier for hole transfer directly from Perovskite to Carbon without an HTL."""
    },
    {
        "id": "Q012",
        "type": "processing",
        "query": """I need to make flexible solar cells on PEN substrates, so processing temperature must stay below 150°C.

Search for low-temperature Electron Transport Layers (ETLs) like SnO2 nanoparticles or ZnOS.
Design a full low-temp device fabrication route.
Predict the PCE and the mechanical flexibility (critical bending radius).
Analyze the interface quality: compare the defect density of perovskite grown on low-temp ETLs versus high-temp sintered TiO2."""
    },
    
    # === Group 5: Frontier Exploration (Q013-Q016) ===
    {
        "id": "Q013",
        "type": "frontier",
        "query": """I heard some perovskites can self-heal after moisture degradation. I want to exploit this for long-life solar cells.

Search for additives (like dynamic polymers or methylamine-complexing agents) that promote self-healing/recrystallization.
Design a material system with this self-healing capability clearly defined.
Predict the PCE recovery percentage after a degradation-healing cycle.
Analyze the chemical reversibility: explain the thermodynamics of the hydration and dehydration reaction involved."""
    },
    {
        "id": "Q014",
        "type": "frontier",
        "query": """We are exploring spintronics and need chiral perovskites.

Search for chiral organic ligands (like R-MBA or S-MBA) incorporated into 2D perovskites recently.
Design a chiral 2D or 2D/3D perovskite structure.
Predict the degree of chirality (CD signal) and the photovoltaic efficiency.
Analyze the mechanism: explain how the chirality transfer occurs from the organic ligand to the inorganic Pb-I framework."""
    },
    {
        "id": "Q015",
        "type": "frontier",
        "query": """There is a theory that ferroelectricity aids charge separation in MAPbI3. I want to enhance this.

Search for evidence or methods to induce ferroelectric domains (e.g., poling, strain engineering).
Design an experiment or recipe to maximize domain alignment.
Predict the impact on the Fill Factor and Short Circuit Current.
Analyze the physics: does the internal electric field from the domains actually assist in dissociating excitons?"""
    },
        {
        "id": "Q016",
        "type": "frontier",
        "query": """We want to use CsPbI3 quantum dots (QDs) for a solar cell, but ligand exchange is tricky.

Search for short-chain ligands aimed at replacing oleic acid to improve conductivity.
Design a layer-by-layer deposition process with a specific ligand exchange solvent.
Predict the carrier mobility and the final PCE.
Analyze the surface defects: how does the new ligand passivation reduce the trap density on the QD surface?"""
    },
    
    # === Group 6: Special Environments & Theory (Q017-Q020) ===
    {
        "id": "Q017",
        "type": "special_environment",
        "query": """We are designing perovskite cells for IoT sensors powered by indoor LED light (1000 lux). The spectrum requires a wide bandgap (~1.9 eV).\n\n1. Search for wide-bandgap perovskite compositions (e.g., CsPbI2Br or FA-based mixed halides) optimized for LED spectra.\n2. Design a recipe that suppresses the halide segregation often seen in Br-rich films.\n3. Predict the output power density (uW/cm²) and the impact of low light intensity on the Fill Factor.\n4. Analyze the loss mechanisms: Explain why high shunt resistance (Rsh) is the critical bottleneck under low-illumination conditions."""
    },
    {
        "id": "Q018",
        "type": "special_environment",
        "query": """We are evaluating perovskite solar cells for space satellites (AM0 spectrum). They must withstand high-energy proton radiation and vacuum.\n\n1. Search for literature comparing the proton radiation hardness of perovskites vs. silicon/GaAs.\n2. Design a device stack (including encapsulation) that utilizes the 'self-healing' defect tolerance of perovskites.\n3. Predict the degradation rate of Voc after a specific fluence of proton irradiation.\n4. Analyze the physics: Does the soft lattice allow for the annihilation of Frenkel defects generated by radiation bombardment?"""
    },
    {
        "id": "Q019",
        "type": "special_environment",
        "query": """Our modules will be deployed in desert environments with extreme day/night temperature swings (thermal cycling from -10°C to 85°C).\n\n1. Search for 'elastic' grain boundary additives (e.g., cross-linkable polymers or elastomers) that can buffer thermal stress.\n2. Design a grain boundary modification strategy to prevent crack propagation during cooling.\n3. Predict the T80 lifetime under IEC 61215 thermal cycling standards.\n4. Analyze the mechanical failure: Discuss the role of Coefficient of Thermal Expansion (CTE) mismatch between the perovskite and the transport layers."""
    },
    {
        "id": "Q020",
        "type": "special_environment",
        "query": """We are developing flexible perovskite cells for wearable electronics that must endure constant bending and stretching.\n\n1. Search for 'scaffold' materials (like porous PU or fibrous networks) that infiltrate the perovskite film to enhance mechanical durability.\n2. Design a flexible device architecture on a polymer substrate (PEN/PET) with a low-temperature process.\n3. Predict the critical bending radius and performance retention after 1000 bending cycles.\n4. Analyze the strain distribution: How does the scaffold dissipate mechanical stress away from the brittle perovskite crystals?"""
    },
    {

        "query":"""
I am currently researching novel perovskite material formulations and need to design experimental protocols that balance three critical conflicting objectives: High Efficiency (PCE > 20%), High Stability (T80 > 1000h), and Reduced Toxicity (Low-Lead or Lead-Free).Please act as a Senior Materials Scientist and execute the following 4-step workflow:Step 1: Literature Review & Trend AnalysisSearch for the most recent high-impact papers (last 3 years) that successfully tackle the 'efficiency-toxicity-stability' trade-off.Identify emerging strategies such as Sn-Pb alloying, Double Perovskites, 2D/3D heterostructures, or Green Solvent Engineering.Step 2: Formula Design (5 Distinct Candidates)Based on Step 1, design 5 distinct perovskite compositions ranging from 'High Performance/Low Lead' to 'Pure Lead-Free'.For each candidate, explicitly state the stoichiometry (e.g., $FA_{0.7}MA_{0.3}Pb_{0.5}Sn_{0.5}I_3$) and the rationale behind its selection.Step 3: Detailed Experimental ProtocolsFor the most promising candidate among the five, provide a step-by-step fabrication protocol covering:Precursor Solution Preparation: Exact molar ratios, solvent choice (prioritize non-toxic solvents like DMSO/Anisole), and mixing conditions.Deposition Method: Detailed spin-coating parameters, anti-solvent quenching timing, and annealing profile.Passivation Strategy: Suggest a specific surface passivation layer (e.g., BAI, PMMA, or molecular additives) to minimize defects.Step 4: Critical Analysis & Trade-off AssessmentCritically analyze each of the 5 designs from Step 2.If a design sacrifices Efficiency for Toxicity (or vice versa), explicitly explain why (e.g., 'Sn-based perovskites have lower Voc due to oxidation' or 'Double perovskites suffer from indirect bandgaps').Provide a theoretical estimation of the expected PCE and Stability limits for each.""" }

    }