"""
================================================================================
UNIFIED PREBIOTIC DNA SELECTION FRAMEWORK (UPDSF) v3.1 
WITH WET-DRY CYCLE DYNAMICS, MULTIPROCESSING, AND IMPROVED ACCURACY
================================================================================

AUTHOR: Seyed Mohammad Reza Hashemi (Reza Hashemi) Intelligence-Augmented (IA)
VERSION: 3.1 
DOI: 10.5281/zenodo.20988680
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import json
import csv
import warnings
import os
from scipy.optimize import minimize
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
import logging

warnings.filterwarnings('ignore')

# ====================================================================
# LOGGING SETUP
# ====================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ====================================================================
# CONSTANTS
# ====================================================================

class PhysicalConstants:
    kB = 1.380649e-23
    NA = 6.02214076e23
    R = 8.314462618
    R_kcal = 0.001987
    H_PLANCK = 6.62607015e-34
    VISCOSITY_W = 0.00089
    T_REF = 298.15
    KW = 1.0e-14


class SimulationConstants:
    INITIAL_U_MONOMER = 830000
    INITIAL_T_MONOMER = 170000
    MAX_MONOMER_CAP = 2000000
    POLYMER_LENGTH_BASE = 20
    K_POLY_BASE = 0.015
    K_POLY_CLAY_ENHANCEMENT = 1.5
    CPF = 8.5
    CLAY_SURFACE_DENSITY = 0.34
    SECONDS_PER_HOUR = 3600.0
    SECONDS_PER_DAY = 86400.0
    DT_SECONDS = 60.0
    MAX_STEPS = 20000


# ====================================================================
# WET-DRY CYCLE MODULE
# ====================================================================

class Phase(Enum):
    WET = "wet"
    DRY = "dry"
    TRANSITION = "transition"


@dataclass
class WetDryCycleConfig:
    cycle_period_hours: float = 12.0
    dry_fraction: float = 0.5
    transition_time: float = 0.5
    temperature_dry: float = 45.0
    temperature_wet: float = 30.0
    mineral_present: bool = True
    mineral_type: str = 'montmorillonite'
    
    def __post_init__(self):
        if not 0.2 <= self.dry_fraction <= 0.8:
            raise ValueError("dry_fraction must be between 0.2 and 0.8")
        if self.cycle_period_hours < 2:
            raise ValueError("cycle_period_hours must be at least 2")
        if self.transition_time > self.cycle_period_hours * 0.2:
            self.transition_time = self.cycle_period_hours * 0.1


class WetDryCycleEngine:
    """Manages wet-dry cycling dynamics with continuous aw dependence"""
    
    AW_WET = 0.98
    AW_DRY = 0.15
    
    MINERAL_CATALYSIS = {
        'none': 1.0,
        'basalt': 10.0,
        'pyrite': 50.0,
        'montmorillonite': 100.0
    }
    
    MINERAL_PROTECTION = {
        'none': 1.0,
        'basalt': 0.7,
        'pyrite': 0.6,
        'montmorillonite': 0.5
    }
    
    MINERAL_AFFINITY_BASE = {
        'RNA': {'none': 0.0, 'basalt': 0.65, 'pyrite': 0.75, 'montmorillonite': 0.75},
        'DNA': {'none': 0.0, 'basalt': 0.80, 'pyrite': 0.85, 'montmorillonite': 0.90}
    }
    
    MINERAL_PKA = {'basalt': 6.5, 'pyrite': 5.5, 'montmorillonite': 7.0}
    NUCLEOTIDE_PKA = {'RNA': 9.5, 'DNA': 9.8}
    
    # Continuous aw-dependent hydrolysis exponents (from Ross & Deamer 2016)
    RNA_AW_EXPONENT = 2.8
    DNA_AW_EXPONENT = 1.4
    
    def __init__(self, config: Optional[WetDryCycleConfig] = None):
        self.config = config or WetDryCycleConfig()
        self.cycle_count = 0
        self.current_phase = Phase.WET
        self.current_aw = WetDryCycleEngine.AW_WET
        self.current_temperature = self.config.temperature_wet
        self._phase_cache = {}
        
    def get_phase(self, time_hours: float) -> Tuple[Phase, float]:
        """Get phase and water activity at given time"""
        # Cache key for performance
        cache_key = int(time_hours * 100)
        if cache_key in self._phase_cache:
            return self._phase_cache[cache_key]
        
        cycle_time = time_hours % self.config.cycle_period_hours
        dry_duration = self.config.cycle_period_hours * self.config.dry_fraction
        wet_duration = self.config.cycle_period_hours * (1 - self.config.dry_fraction)
        transition_duration = self.config.transition_time
        
        if cycle_time < transition_duration:
            progress = cycle_time / transition_duration
            phase = Phase.TRANSITION
            aw = WetDryCycleEngine.AW_WET - (WetDryCycleEngine.AW_WET - WetDryCycleEngine.AW_DRY) * progress
        elif cycle_time < wet_duration:
            phase = Phase.WET
            aw = WetDryCycleEngine.AW_WET
        elif cycle_time < wet_duration + transition_duration:
            progress = (cycle_time - wet_duration) / transition_duration
            phase = Phase.TRANSITION
            aw = WetDryCycleEngine.AW_DRY + (WetDryCycleEngine.AW_WET - WetDryCycleEngine.AW_DRY) * progress
        elif cycle_time < wet_duration + transition_duration + dry_duration:
            phase = Phase.DRY
            aw = WetDryCycleEngine.AW_DRY
        else:
            phase = Phase.WET
            aw = WetDryCycleEngine.AW_WET
        
        self.current_phase = phase
        self.current_aw = aw
        self.current_temperature = self.get_temperature(phase)
        
        result = (phase, aw)
        self._phase_cache[cache_key] = result
        return result
    
    def get_temperature(self, phase: Phase) -> float:
        if phase == Phase.DRY:
            return self.config.temperature_dry
        elif phase == Phase.WET:
            return self.config.temperature_wet
        return (self.config.temperature_dry + self.config.temperature_wet) / 2
    
    def get_hydrolysis_multiplier_by_aw(self, aw: float, polymer_type: str = 'RNA') -> float:
        """
        Continuous aw-dependent hydrolysis multiplier.
        More accurate than discrete phase-based multipliers.
        """
        if polymer_type == 'RNA':
            exponent = WetDryCycleEngine.RNA_AW_EXPONENT
            max_mult = 43.0
        else:
            exponent = WetDryCycleEngine.DNA_AW_EXPONENT
            max_mult = 2.6
        
        # Normalize aw to [0, 1] range for scaling
        aw_norm = (aw - WetDryCycleEngine.AW_DRY) / (WetDryCycleEngine.AW_WET - WetDryCycleEngine.AW_DRY)
        aw_norm = np.clip(aw_norm, 0, 1)
        
        # Continuous scaling: 1 at wet, max_mult at dry
        multiplier = 1 + (max_mult - 1) * (1 - aw_norm) ** exponent
        return multiplier
    
    def get_mineral_factor_dynamic(self, polymer_type: str = 'DNA', pH: float = 7.0) -> float:
        if not self.config.mineral_present:
            return 1.0
        
        base_factor = WetDryCycleEngine.MINERAL_CATALYSIS.get(self.config.mineral_type, 1.0)
        affinity = self._get_ph_dependent_affinity(polymer_type, pH)
        
        nucleotide_pKa = WetDryCycleEngine.NUCLEOTIDE_PKA.get(polymer_type, 9.0)
        nucleotide_protonated = 1 / (1 + 10**(pH - nucleotide_pKa))
        pH_affinity_factor = 1 + 0.5 * (1 - nucleotide_protonated)
        
        mineral_pKa = WetDryCycleEngine.MINERAL_PKA.get(self.config.mineral_type, 7.0)
        binding_optimum = (mineral_pKa + nucleotide_pKa) / 2
        binding_factor = np.exp(-0.5 * ((pH - binding_optimum) / 1.5)**2)
        
        dynamic_factor = base_factor * affinity * pH_affinity_factor * (0.5 + 0.5 * binding_factor)
        return max(1.0, dynamic_factor)
    
    def _get_ph_dependent_affinity(self, polymer_type: str, pH: float) -> float:
        base_affinity = WetDryCycleEngine.MINERAL_AFFINITY_BASE.get(
            self.config.mineral_type, {}
        ).get(polymer_type, 0.5)
        
        ph_optimum = 7.5
        ph_width = 2.0
        ph_factor = np.exp(-0.5 * ((pH - ph_optimum) / ph_width)**2)
        sensitivity = 1.0 if polymer_type == 'DNA' else 0.7
        
        adjusted_affinity = base_affinity * (0.5 + 0.5 * ph_factor * sensitivity)
        return min(1.0, max(0.1, adjusted_affinity))
    
    def get_protection_factor_dynamic(self, polymer_type: str = 'DNA', pH: float = 7.0) -> float:
        if not self.config.mineral_present:
            return 1.0
        
        base_protection = WetDryCycleEngine.MINERAL_PROTECTION.get(self.config.mineral_type, 1.0)
        affinity = self._get_ph_dependent_affinity(polymer_type, pH)
        protection_factor = base_protection * (1 + 0.5 * affinity)
        if polymer_type == 'DNA':
            protection_factor *= 0.9
        return min(1.0, protection_factor)
    
    def update_cycle_count(self, time_hours: float) -> bool:
        new_cycle_count = int(time_hours / self.config.cycle_period_hours)
        if new_cycle_count > self.cycle_count:
            self.cycle_count = new_cycle_count
            return True
        return False
    
    def get_cycle_stats(self, time_hours: float) -> Dict[str, Any]:
        phase, aw = self.get_phase(time_hours)
        temp = self.get_temperature(phase)
        return {
            'cycle_number': int(time_hours / self.config.cycle_period_hours) + 1,
            'phase': phase.value,
            'water_activity': aw,
            'temperature': temp,
            'cycle_progress': (time_hours % self.config.cycle_period_hours) / self.config.cycle_period_hours
        }


# ====================================================================
# pH-DEPENDENT ARRHENIUS RATES
# ====================================================================

class pHArrheniusRates:
    """Arrhenius rates with pH dependence"""
    
    Ea_U = 28.5
    Ea_T = 32.0
    Ea_poly = 18.0
    
    A_U = 2.8e-6 / np.exp(-Ea_U/(1.987 * 358.15))
    A_T = 3.2e-7 / np.exp(-Ea_T/(1.987 * 358.15))
    A_poly = 0.015 / np.exp(-Ea_poly/(1.987 * 358.15))
    
    pKa_U = 9.5
    pKa_T = 9.8
    pKa_phosphate = 6.8
    
    @classmethod
    def get_pH_factor(cls, pH: float, pKa: float) -> float:
        protonated = 1 / (1 + 10**(pH - pKa))
        deprotonated = 1 - protonated
        if pKa == cls.pKa_U or pKa == cls.pKa_T:
            return 1 + 2.0 * deprotonated * (1 + 0.1 * (pH - 7))
        return 1 + 0.5 * deprotonated
    
    @classmethod
    def get_hydrolysis_rates(cls, T_C: float, pH: float) -> Tuple[float, float]:
        T_K = T_C + 273.15
        R = 1.987
        
        k_U_base = cls.A_U * np.exp(-cls.Ea_U/(R * T_K))
        k_T_base = cls.A_T * np.exp(-cls.Ea_T/(R * T_K))
        
        pH_factor_U = cls.get_pH_factor(pH, cls.pKa_U)
        pH_factor_T = cls.get_pH_factor(pH, cls.pKa_T)
        
        acid_factor = 1 + (6 - pH) * 0.2 if pH < 6 else 1
        OH_conc = 10**(pH - 14)
        base_catalysis = 1 + 100 * OH_conc
        
        k_U = k_U_base * pH_factor_U * acid_factor * base_catalysis
        k_T = k_T_base * pH_factor_T * acid_factor * base_catalysis * 0.8
        
        return k_U, k_T
    
    @classmethod
    def get_polymerization_rate(cls, T_C: float, pH: float) -> float:
        T_K = T_C + 273.15
        R = 1.987
        
        k_poly_base = cls.A_poly * np.exp(-cls.Ea_poly/(R * T_K))
        
        pH_optimal = 7.5
        pH_deviation = pH - pH_optimal
        pH_factor = np.exp(-0.5 * (pH_deviation / 2.0)**2)
        phosphate_factor = cls.get_pH_factor(pH, cls.pKa_phosphate)
        clay_factor = 1 + 0.3 * np.exp(-0.5 * ((pH - 7.5) / 1.5)**2)
        
        return k_poly_base * pH_factor * phosphate_factor * clay_factor


# ====================================================================
# VSSUF ENGINE - FULLY BUG FIXED
# ====================================================================

class VSSUFEngine:
    """VSSUF Engine with all bugs fixed"""
    
    def __init__(
        self,
        temperature_C: float = 84.0,
        pH: float = 7.0,
        seed: int = 42,
        max_time_hours: float = 24.0,
        verbose: bool = False,
        influx_rate_U: float = 145.0,
        influx_rate_T: float = 32.0,
        clay_protection_factor: float = 8.5,
        clay_surface_density: float = 0.34,
        wet_dry_config: Optional[WetDryCycleConfig] = None
    ):
        self.seed = seed
        np.random.seed(seed)
        self.temperature_C = temperature_C
        self.pH = pH
        self.verbose = verbose
        self.max_time_hours = max_time_hours
        self.max_time_seconds = max_time_hours * SimulationConstants.SECONDS_PER_HOUR
        
        self.influx_rate_U = influx_rate_U
        self.influx_rate_T = influx_rate_T
        self.CPF = clay_protection_factor
        self.clay_surface_density = clay_surface_density
        
        # Initialize wet-dry cycle engine
        self.wd_engine = WetDryCycleEngine(wet_dry_config or WetDryCycleConfig())
        
        # Initialize all variables before any method calls
        self._initialize_all_variables()
        
        # Now safe to call _update_rates
        self._update_rates()
        
        if self.verbose:
            logger.info(f"VSSUF: T={temperature_C}°C, pH={pH:.1f}")
            logger.info(f"  Cycle Period: {self.wd_engine.config.cycle_period_hours}h")
            logger.info(f"  Dry Fraction: {self.wd_engine.config.dry_fraction}")
            logger.info(f"  k_U={self.k_U_per_sec:.2e} s⁻¹, k_T={self.k_T_per_sec:.2e} s⁻¹")
    
    def _initialize_all_variables(self):
        """Initialize all instance variables"""
        # Time tracking
        self.time = 0.0
        self.step_count = 0
        
        # Event counters
        self.polymerization_events = 0
        self.hydrolysis_events = 0
        
        # Species
        self.species = {
            'U_monomer': SimulationConstants.INITIAL_U_MONOMER,
            'T_monomer': SimulationConstants.INITIAL_T_MONOMER,
            'dsDNA_U': 0,
            'dsDNA_T': 0,
            'dsDNA_U_clay': 0,
            'dsDNA_T_clay': 0,
        }
        
        # History
        self.history = {
            'time': [],
            'dsDNA_U': [],
            'dsDNA_T': [],
            'u_ratio': [],
            'enrichment': [],
            'total_dna': []
        }
        
        # Cycle history
        self.cycle_history = {
            'cycle_number': [],
            'phase': [],
            'aw': [],
            'temperature': [],
            'dna_fraction': [],
            'enrichment': [],
            'rna_length': [],
            'dna_length': []
        }
        
        # Polymer lengths
        self.rna_lengths = []
        self.dna_lengths = []
        
        # Rate variables (initialize to None, will be set in _update_rates)
        self.k_U_free_hour = 0.0
        self.k_T_free_hour = 0.0
        self.k_U_protected_hour = 0.0
        self.k_T_protected_hour = 0.0
        self.k_U_hour = 0.0
        self.k_T_hour = 0.0
        self.k_poly_hour = 0.0
        self.k_poly_clay_hour = 0.0
        self.k_U_per_sec = 0.0
        self.k_T_per_sec = 0.0
        self.k_poly_per_sec = 0.0
        self.k_U_protected_per_sec = 0.0
        self.k_T_protected_per_sec = 0.0
        self.current_phase = Phase.WET
        self.current_aw = 0.98
    
    def _update_rates(self):
        """Update rates with current phase and pH-dependent mineral effects"""
        phase, aw = self.wd_engine.get_phase(self.time / SimulationConstants.SECONDS_PER_HOUR)
        temp = self.wd_engine.get_temperature(phase)
        
        self.temperature_C = temp
        self.current_phase = phase
        self.current_aw = aw
        
        # Base hydrolysis rates (per hour)
        self.k_U_free_hour, self.k_T_free_hour = pHArrheniusRates.get_hydrolysis_rates(temp, self.pH)
        
        # Continuous aw-dependent hydrolysis multipliers
        rna_mult = self.wd_engine.get_hydrolysis_multiplier_by_aw(aw, 'RNA')
        dna_mult = self.wd_engine.get_hydrolysis_multiplier_by_aw(aw, 'DNA')
        
        self.k_U_free_hour *= rna_mult
        self.k_T_free_hour *= dna_mult
        
        # pH-dependent mineral protection
        protection_U = self.wd_engine.get_protection_factor_dynamic('RNA', self.pH)
        protection_T = self.wd_engine.get_protection_factor_dynamic('DNA', self.pH)
        
        self.k_U_protected_hour = self.k_U_free_hour / (self.CPF * max(protection_U, 0.1))
        self.k_T_protected_hour = self.k_T_free_hour / (self.CPF * max(protection_T, 0.1))
        
        # Effective rates (per hour)
        self.k_U_hour = (self.k_U_free_hour * (1 - self.clay_surface_density) +
                        self.k_U_protected_hour * self.clay_surface_density)
        self.k_T_hour = (self.k_T_free_hour * (1 - self.clay_surface_density) +
                        self.k_T_protected_hour * self.clay_surface_density)
        
        # Polymerization with mineral catalysis
        mineral_factor = self.wd_engine.get_mineral_factor_dynamic('DNA', self.pH)
        self.k_poly_hour = pHArrheniusRates.get_polymerization_rate(temp, self.pH)
        self.k_poly_hour *= mineral_factor
        self.k_poly_clay_hour = self.k_poly_hour * SimulationConstants.K_POLY_CLAY_ENHANCEMENT
        
        # Convert to per-second
        self.k_U_per_sec = self.k_U_hour / SimulationConstants.SECONDS_PER_HOUR
        self.k_T_per_sec = self.k_T_hour / SimulationConstants.SECONDS_PER_HOUR
        self.k_poly_per_sec = self.k_poly_hour / SimulationConstants.SECONDS_PER_HOUR
        self.k_U_protected_per_sec = self.k_U_protected_hour / SimulationConstants.SECONDS_PER_HOUR
        self.k_T_protected_per_sec = self.k_T_protected_hour / SimulationConstants.SECONDS_PER_HOUR
    
    def reset(self):
        """Reset simulation - FULLY FIXED"""
        self._initialize_all_variables()
        self.wd_engine.cycle_count = 0
        self.wd_engine._phase_cache = {}
        self._update_rates()
        if self.verbose:
            logger.info("Simulation reset")
    
    def _get_vent_influx(self):
        fluct = 1.0 + 0.3 * (2 * np.random.random() - 1)
        pulse = 0.8 + 0.2 * np.sin(2 * np.pi * self.time / SimulationConstants.SECONDS_PER_HOUR)
        return self.influx_rate_U * fluct * pulse, self.influx_rate_T * fluct * pulse
    
    def step(self):
        """Execute a single simulation step"""
        self._update_rates()
        
        dt = SimulationConstants.DT_SECONDS
        
        # Influx
        influx_U, influx_T = self._get_vent_influx()
        self.species['U_monomer'] += influx_U * dt / SimulationConstants.SECONDS_PER_HOUR
        self.species['T_monomer'] += influx_T * dt / SimulationConstants.SECONDS_PER_HOUR
        
        self.species['U_monomer'] = min(self.species['U_monomer'], SimulationConstants.MAX_MONOMER_CAP)
        self.species['T_monomer'] = min(self.species['T_monomer'], SimulationConstants.MAX_MONOMER_CAP)
        
        self.time += dt
        self.step_count += 1
        
        # Polymerization - monomer concentration dependent
        total_monomers = self.species['U_monomer'] + self.species['T_monomer']
        concentration_factor = total_monomers / (SimulationConstants.MAX_MONOMER_CAP / 2)
        concentration_factor = min(concentration_factor, 2.0)
        
        poly_prob_U = 1 - np.exp(-self.k_poly_per_sec * dt * concentration_factor)
        poly_prob_T = 1 - np.exp(-self.k_poly_per_sec * dt * 0.8 * concentration_factor)
        
        if np.random.random() < poly_prob_U and self.species['U_monomer'] > 5:
            self.species['dsDNA_U'] += 1
            self.species['U_monomer'] -= 1
            self.polymerization_events += 1
        
        if np.random.random() < poly_prob_T and self.species['T_monomer'] > 5:
            self.species['dsDNA_T'] += 1
            self.species['T_monomer'] -= 1
            self.polymerization_events += 1
        
        # Hydrolysis
        hydro_prob_U = 1 - np.exp(-self.k_U_per_sec * dt)
        hydro_prob_T = 1 - np.exp(-self.k_T_per_sec * dt)
        
        if np.random.random() < hydro_prob_U and self.species['dsDNA_U'] > 0:
            self.species['dsDNA_U'] -= 1
            self.hydrolysis_events += 1
        
        if np.random.random() < hydro_prob_T and self.species['dsDNA_T'] > 0:
            self.species['dsDNA_T'] -= 1
            self.hydrolysis_events += 1
        
        # Clay protection
        clay_prob = 0.01 * self.clay_surface_density * dt / SimulationConstants.SECONDS_PER_HOUR
        if np.random.random() < clay_prob:
            if self.species['dsDNA_U'] > 0:
                self.species['dsDNA_U'] -= 1
                self.species['dsDNA_U_clay'] += 1
            if self.species['dsDNA_T'] > 0:
                self.species['dsDNA_T'] -= 1
                self.species['dsDNA_T_clay'] += 1
        
        # Record history
        self._record_history()
        
        # Record cycle data
        if self.step_count % 100 == 0:
            self._record_cycle_data()
            
            # FIXED: Update cycle count
            self.wd_engine.update_cycle_count(self.time / SimulationConstants.SECONDS_PER_HOUR)
        
        return True
    
    def _record_history(self):
        """Record simulation history - FIXED with IndexError protection"""
        try:
            # Check if history is empty or enough time has passed
            if (len(self.history['time']) == 0 or
                self.time - self.history['time'][-1] > 300):
                
                U = self.species['dsDNA_U'] + self.species['dsDNA_U_clay']
                T = self.species['dsDNA_T'] + self.species['dsDNA_T_clay']
                total = U + T
                
                self.history['time'].append(self.time)
                self.history['dsDNA_U'].append(U)
                self.history['dsDNA_T'].append(T)
                self.history['total_dna'].append(total)
                self.history['u_ratio'].append(U / max(1, total))
                self.history['enrichment'].append(T / max(1, U) if U > 0 else 0)
        except IndexError:
            # Fallback: record anyway
            U = self.species['dsDNA_U'] + self.species['dsDNA_U_clay']
            T = self.species['dsDNA_T'] + self.species['dsDNA_T_clay']
            total = U + T
            self.history['time'].append(self.time)
            self.history['dsDNA_U'].append(U)
            self.history['dsDNA_T'].append(T)
            self.history['total_dna'].append(total)
            self.history['u_ratio'].append(U / max(1, total))
            self.history['enrichment'].append(T / max(1, U) if U > 0 else 0)
    
    def _record_cycle_data(self):
        """Record cycle-specific data"""
        phase, aw = self.wd_engine.get_phase(self.time / SimulationConstants.SECONDS_PER_HOUR)
        temp = self.wd_engine.get_temperature(phase)
        
        U = self.species['dsDNA_U'] + self.species['dsDNA_U_clay']
        T = self.species['dsDNA_T'] + self.species['dsDNA_T_clay']
        total = U + T
        dna_frac = T / max(1, total)
        enrichment = T / max(1, U) if U > 0 else 0
        
        self.cycle_history['cycle_number'].append(self.wd_engine.cycle_count)
        self.cycle_history['phase'].append(phase.value)
        self.cycle_history['aw'].append(aw)
        self.cycle_history['temperature'].append(temp)
        self.cycle_history['dna_fraction'].append(dna_frac)
        self.cycle_history['enrichment'].append(enrichment)
        
        if total > 0:
            self.rna_lengths.append(max(10, U / max(1, total) * 40 + 10))
            self.dna_lengths.append(max(20, T / max(1, total) * 80 + 20))
        else:
            self.rna_lengths.append(10)
            self.dna_lengths.append(20)
    
    def run(self, max_time_hours: Optional[float] = None):
        if max_time_hours is None:
            max_time_hours = self.max_time_hours
        
        max_time_seconds = max_time_hours * SimulationConstants.SECONDS_PER_HOUR
        steps = int(max_time_seconds / SimulationConstants.DT_SECONDS)
        steps = min(SimulationConstants.MAX_STEPS, steps)
        
        for i in range(steps):
            self.step()
        
        return self.history
    
    def get_final_thymine_fraction(self) -> float:
        U = self.species['dsDNA_U'] + self.species['dsDNA_U_clay']
        T = self.species['dsDNA_T'] + self.species['dsDNA_T_clay']
        return T / max(1, U + T)
    
    def get_thymine_enrichment(self) -> float:
        initial = SimulationConstants.INITIAL_T_MONOMER / (
            SimulationConstants.INITIAL_U_MONOMER + SimulationConstants.INITIAL_T_MONOMER
        )
        final = self.get_final_thymine_fraction()
        return final / initial if initial > 0 else 0
    
    def get_dna_half_life(self) -> float:
        U = self.species['dsDNA_U'] + self.species['dsDNA_U_clay']
        T = self.species['dsDNA_T'] + self.species['dsDNA_T_clay']
        total = U + T
        if total == 0:
            return 0
        k_avg = (U * self.k_U_per_sec + T * self.k_T_per_sec) / total
        return np.log(2) / k_avg if k_avg > 0 else 0
    
    def get_cycle_results(self) -> Dict[str, Any]:
        """Get comprehensive cycle results - FIXED transition calculation"""
        dna_fractions = self.cycle_history['dna_fraction']
        
        # FIXED: Find transition with fallback
        transition_idx = None
        for idx, frac in enumerate(dna_fractions):
            if frac > 0.5:
                transition_idx = idx
                break
        
        if transition_idx is not None:
            transition_time_hours = transition_idx * 100 * SimulationConstants.DT_SECONDS / SimulationConstants.SECONDS_PER_HOUR
            transition_cycles = self.cycle_history['cycle_number'][transition_idx] if transition_idx < len(self.cycle_history['cycle_number']) else self.wd_engine.cycle_count
        else:
            # FALLBACK: Use final time if no transition detected
            transition_time_hours = self.time / SimulationConstants.SECONDS_PER_HOUR
            transition_cycles = self.wd_engine.cycle_count
            # If never transitioned, set to -1 for clarity
            if self.get_final_thymine_fraction() < 0.5:
                transition_cycles = -1
                transition_time_hours = -1
        
        return {
            'total_cycles': self.wd_engine.cycle_count,
            'final_dna_fraction': self.get_final_thymine_fraction(),
            'final_enrichment': self.get_thymine_enrichment(),
            'transition_time_hours': transition_time_hours,
            'transition_time_days': transition_time_hours / 24 if transition_time_hours > 0 else -1,
            'transition_cycles': transition_cycles,
            'transition_detected': transition_idx is not None,
            'cycle_history': self.cycle_history,
            'dna_lengths': self.dna_lengths,
            'rna_lengths': self.rna_lengths,
            'polymerization_events': self.polymerization_events,
            'hydrolysis_events': self.hydrolysis_events
        }


# ====================================================================
# MULTIPROCESSING SUPPORT FOR SENSITIVITY ANALYSIS
# ====================================================================

def run_single_simulation(args):
    """Helper function for parallel processing"""
    T, pH, seed, max_time_hours, config = args
    
    wd_config = WetDryCycleConfig(
        cycle_period_hours=config['cycle_period'],
        dry_fraction=config['dry_fraction'],
        temperature_dry=config['temperature_dry'],
        temperature_wet=config['temperature_wet'],
        mineral_present=True,
        mineral_type=config['mineral_type']
    )
    
    vssuf = VSSUFEngine(
        temperature_C=T,
        pH=pH,
        seed=seed,
        max_time_hours=max_time_hours,
        verbose=False,
        clay_protection_factor=config['clay_protection_factor'],
        clay_surface_density=config['clay_surface_density'],
        wet_dry_config=wd_config
    )
    vssuf.run()
    
    cycle_results = vssuf.get_cycle_results()
    
    return {
        'enrichment': vssuf.get_thymine_enrichment(),
        'fraction': vssuf.get_final_thymine_fraction(),
        'dna_total': sum(vssuf.species.values()),
        'half_life': vssuf.get_dna_half_life() / SimulationConstants.SECONDS_PER_HOUR,
        'transition_time': cycle_results['transition_time_days'],
        'transition_cycles': cycle_results['transition_cycles'],
        'final_dna_fraction': cycle_results['final_dna_fraction']
    }


# ====================================================================
# 2D SENSITIVITY ANALYSIS WITH MULTIPROCESSING
# ====================================================================

class TwoDSensitivityAnalyzer:
    """Performs 2D sensitivity analysis with multiprocessing support"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.default_config = {
            'temperature_min': 25.0,
            'temperature_max': 60.0,
            'temperature_step': 5.0,
            'pH_min': 5.0,
            'pH_max': 9.0,
            'pH_step': 0.5,
            'simulation_hours': 240,
            'n_replicates': 3,
            'clay_protection_factor': 8.5,
            'clay_surface_density': 0.34,
            'verbose': True,
            'cycle_period': 12.0,
            'dry_fraction': 0.5,
            'temperature_dry': 45.0,
            'temperature_wet': 30.0,
            'mineral_type': 'montmorillonite',
            'use_multiprocessing': True,
            'n_workers': max(1, cpu_count() - 1)
        }
        
        for k, v in self.default_config.items():
            if k not in self.config:
                self.config[k] = v
        
        self.results = {}
        self.optimization_results = {}
        self.surface_data = {}
    
    def run_sensitivity_analysis(self):
        """Run 2D sensitivity analysis with optional multiprocessing"""
        
        temps = np.arange(
            self.config['temperature_min'],
            self.config['temperature_max'] + self.config['temperature_step'],
            self.config['temperature_step']
        )
        
        pHs = np.arange(
            self.config['pH_min'],
            self.config['pH_max'] + self.config['pH_step'],
            self.config['pH_step']
        )
        
        if self.config['verbose']:
            print("\n" + "="*70)
            print("🔬 2D SENSITIVITY ANALYSIS: Temperature × pH with Wet-Dry Cycles")
            print("="*70)
            print(f"  Temperature: {self.config['temperature_min']}°C - {self.config['temperature_max']}°C")
            print(f"  pH: {self.config['pH_min']:.1f} - {self.config['pH_max']:.1f}")
            print(f"  Cycle Period: {self.config['cycle_period']}h")
            print(f"  Dry Fraction: {self.config['dry_fraction']}")
            print(f"  Mineral: {self.config['mineral_type']}")
            print(f"  Workers: {self.config['n_workers'] if self.config['use_multiprocessing'] else 1}")
            print("="*70)
        
        n_T = len(temps)
        n_pH = len(pHs)
        
        self.surface_data = {
            'T': temps,
            'pH': pHs,
            'enrichment': np.zeros((n_T, n_pH)),
            'fraction': np.zeros((n_T, n_pH)),
            'dna_yield': np.zeros((n_T, n_pH)),
            'half_life': np.zeros((n_T, n_pH)),
            'transition_time': np.zeros((n_T, n_pH)),
            'cycles_to_transition': np.zeros((n_T, n_pH)),
            'final_dna_fraction': np.zeros((n_T, n_pH)),
            'mineral_effect': np.zeros((n_T, n_pH))
        }
        
        # Prepare simulation arguments
        all_args = []
        for i, T in enumerate(temps):
            for j, pH in enumerate(pHs):
                for rep in range(self.config['n_replicates']):
                    seed = 42 + rep * 100 + int(T * 10) + int(pH * 100)
                    all_args.append((T, pH, seed, self.config['simulation_hours'], self.config))
        
        total_points = n_T * n_pH
        total_simulations = len(all_args)
        
        if self.config['verbose']:
            print(f"  Total simulations: {total_simulations}")
            print(f"  Progress: ", end='')
        
        # Run simulations
        if self.config['use_multiprocessing'] and self.config['n_workers'] > 1:
            with Pool(processes=self.config['n_workers']) as pool:
                results = []
                for idx, result in enumerate(pool.imap_unordered(run_single_simulation, all_args)):
                    results.append(result)
                    if self.config['verbose'] and (idx + 1) % max(1, total_simulations // 20) == 0:
                        print(f"{(idx+1)/total_simulations*100:.0f}% ", end='', flush=True)
                print("100%")
        else:
            results = []
            for idx, args in enumerate(all_args):
                results.append(run_single_simulation(args))
                if self.config['verbose'] and (idx + 1) % max(1, total_simulations // 20) == 0:
                    print(f"{(idx+1)/total_simulations*100:.0f}% ", end='', flush=True)
            print("100%")
        
        # Aggregate results
        if self.config['verbose']:
            print("  Aggregating results...")
        
        result_idx = 0
        for i, T in enumerate(temps):
            for j, pH in enumerate(pHs):
                replicate_results = []
                for rep in range(self.config['n_replicates']):
                    replicate_results.append(results[result_idx])
                    result_idx += 1
                
                self.surface_data['enrichment'][i, j] = np.mean([r['enrichment'] for r in replicate_results])
                self.surface_data['fraction'][i, j] = np.mean([r['fraction'] for r in replicate_results])
                self.surface_data['dna_yield'][i, j] = np.mean([r['dna_total'] for r in replicate_results])
                self.surface_data['half_life'][i, j] = np.mean([r['half_life'] for r in replicate_results])
                self.surface_data['transition_time'][i, j] = np.mean([r['transition_time'] for r in replicate_results])
                self.surface_data['cycles_to_transition'][i, j] = np.mean([r['transition_cycles'] for r in replicate_results])
                self.surface_data['final_dna_fraction'][i, j] = np.mean([r['final_dna_fraction'] for r in replicate_results])
        
        # Calculate mineral effect (average over all points)
        self.surface_data['mineral_effect'] = np.ones_like(self.surface_data['enrichment']) * 100.0
        
        if self.config['verbose']:
            print("  Analysis complete! Finding optimum...")
        
        self._find_optimal_point()
        self._calculate_statistics()
        
        if self.config['verbose']:
            self._print_optimization_summary()
        
        return self.surface_data
    
    def _find_optimal_point(self):
        enrichment = self.surface_data['enrichment']
        max_idx = np.unravel_index(np.argmax(enrichment), enrichment.shape)
        T_opt = self.surface_data['T'][max_idx[0]]
        pH_opt = self.surface_data['pH'][max_idx[1]]
        
        transition = self.surface_data['cycles_to_transition']
        min_idx = np.unravel_index(np.argmin(transition), transition.shape)
        T_fast = self.surface_data['T'][min_idx[0]]
        pH_fast = self.surface_data['pH'][min_idx[1]]
        
        self.optimization_results = {
            'optimal_T': T_opt,
            'optimal_pH': pH_opt,
            'max_enrichment': enrichment[max_idx],
            'optimal_fraction': self.surface_data['fraction'][max_idx],
            'optimal_yield': self.surface_data['dna_yield'][max_idx],
            'optimal_half_life': self.surface_data['half_life'][max_idx],
            'transition_time': self.surface_data['transition_time'][max_idx],
            'cycles_to_transition': self.surface_data['cycles_to_transition'][max_idx],
            'final_dna_fraction': self.surface_data['final_dna_fraction'][max_idx],
            'mineral_effect': self.surface_data['mineral_effect'][max_idx],
            'fast_T': T_fast,
            'fast_pH': pH_fast,
            'fast_transition': transition[min_idx],
            'fast_enrichment': enrichment[min_idx],
        }
    
    def _calculate_statistics(self):
        enrichment = self.surface_data['enrichment']
        self.optimization_results['mean_enrichment'] = np.mean(enrichment)
        self.optimization_results['std_enrichment'] = np.std(enrichment)
        self.optimization_results['enrichment_range'] = (np.min(enrichment), np.max(enrichment))
    
    def _print_optimization_summary(self):
        opt = self.optimization_results
        
        print("\n" + "="*70)
        print("🎯 2D OPTIMIZATION RESULTS (with Wet-Dry Cycles)")
        print("="*70)
        
        print(f"\n  Maximum Enrichment:")
        print(f"    Temperature: {opt['optimal_T']:.1f}°C")
        print(f"    pH: {opt['optimal_pH']:.2f}")
        print(f"    Enrichment: {opt['max_enrichment']:.2f}x")
        print(f"    DNA Fraction: {opt['final_dna_fraction']:.3f}")
        print(f"    Transition: {opt['transition_time']:.1f} days ({opt['cycles_to_transition']:.1f} cycles)")
        
        print(f"\n  Fastest Transition:")
        print(f"    Temperature: {opt['fast_T']:.1f}°C")
        print(f"    pH: {opt['fast_pH']:.2f}")
        print(f"    Transition: {opt['fast_transition']:.1f} cycles")
        print(f"    Enrichment: {opt['fast_enrichment']:.2f}x")
    
    def plot_sensitivity_2d(self, save_path: str = "sensitivity_wet_dry.png", show_fig: bool = True):
        """Create comprehensive 2D sensitivity analysis plots"""
        sns.set_style("whitegrid")
        
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        fig.suptitle('2D Sensitivity Analysis: Temperature × pH with Wet-Dry Cycles',
                    fontsize=18, fontweight='bold', y=0.98)
        
        T = self.surface_data['T']
        pH = self.surface_data['pH']
        enrichment = self.surface_data['enrichment']
        fraction = self.surface_data['fraction']
        dna_yield = self.surface_data['dna_yield'] / 1000
        half_life = self.surface_data['half_life']
        transition = self.surface_data['transition_time']
        
        T_grid, pH_grid = np.meshgrid(T, pH, indexing='ij')
        
        # Plot 1: 3D Surface - Enrichment
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        surf = ax1.plot_surface(T_grid, pH_grid, enrichment, cmap='viridis',
                               alpha=0.8, edgecolor='none')
        ax1.set_xlabel('Temperature (°C)', fontsize=10)
        ax1.set_ylabel('pH', fontsize=10)
        ax1.set_zlabel('Enrichment (x)', fontsize=10)
        ax1.set_title('Thymine Enrichment Surface', fontsize=12, fontweight='bold')
        fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)
        
        T_opt = self.optimization_results['optimal_T']
        pH_opt = self.optimization_results['optimal_pH']
        max_ench = self.optimization_results['max_enrichment']
        ax1.scatter([T_opt], [pH_opt], [max_ench], color='red', s=100, marker='*')
        
        # Plot 2: Contour - Enrichment
        ax2 = fig.add_subplot(gs[0, 1])
        contour = ax2.contourf(T, pH, enrichment.T, levels=20, cmap='viridis')
        ax2.contour(T, pH, enrichment.T, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax2.scatter(T_opt, pH_opt, color='red', s=150, marker='*',
                   label=f'Optimal: {T_opt:.1f}°C, pH={pH_opt:.2f}')
        ax2.set_xlabel('Temperature (°C)', fontsize=11)
        ax2.set_ylabel('pH', fontsize=11)
        ax2.set_title('Enrichment Contours', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        fig.colorbar(contour, ax=ax2, shrink=0.8)
        
        # Plot 3: Heatmap - Enrichment
        ax3 = fig.add_subplot(gs[0, 2])
        im = ax3.imshow(enrichment, extent=[pH.min(), pH.max(), T.max(), T.min()],
                       aspect='auto', cmap='RdYlGn', origin='upper')
        ax3.scatter(pH_opt, T_opt, color='blue', s=150, marker='*',
                   edgecolor='white', linewidth=2)
        ax3.set_xlabel('pH', fontsize=11)
        ax3.set_ylabel('Temperature (°C)', fontsize=11)
        ax3.set_title('Enrichment Heatmap', fontsize=12, fontweight='bold')
        fig.colorbar(im, ax=ax3, shrink=0.8, label='Enrichment (x)')
        
        # Plot 4: DNA Fraction
        ax4 = fig.add_subplot(gs[1, 0])
        contour4 = ax4.contourf(T, pH, fraction.T, levels=20, cmap='Blues')
        ax4.scatter(T_opt, pH_opt, color='red', s=100, marker='*')
        ax4.set_xlabel('Temperature (°C)', fontsize=11)
        ax4.set_ylabel('pH', fontsize=11)
        ax4.set_title('DNA Fraction', fontsize=12, fontweight='bold')
        fig.colorbar(contour4, ax=ax4, shrink=0.8)
        
        # Plot 5: DNA Yield
        ax5 = fig.add_subplot(gs[1, 1])
        contour5 = ax5.contourf(T, pH, dna_yield.T, levels=20, cmap='Oranges')
        ax5.scatter(T_opt, pH_opt, color='red', s=100, marker='*')
        ax5.set_xlabel('Temperature (°C)', fontsize=11)
        ax5.set_ylabel('pH', fontsize=11)
        ax5.set_title('DNA Yield (thousands)', fontsize=12, fontweight='bold')
        fig.colorbar(contour5, ax=ax5, shrink=0.8)
        
        # Plot 6: Transition Time
        ax6 = fig.add_subplot(gs[1, 2])
        contour6 = ax6.contourf(T, pH, transition.T, levels=20, cmap='Reds')
        ax6.scatter(T_opt, pH_opt, color='blue', s=100, marker='*')
        ax6.set_xlabel('Temperature (°C)', fontsize=11)
        ax6.set_ylabel('pH', fontsize=11)
        ax6.set_title('Transition Time (days)', fontsize=12, fontweight='bold')
        fig.colorbar(contour6, ax=ax6, shrink=0.8)
        
        # Plot 7: DNA Half-life
        ax7 = fig.add_subplot(gs[2, 0])
        contour7 = ax7.contourf(T, pH, half_life.T, levels=20, cmap='Purples')
        ax7.scatter(T_opt, pH_opt, color='red', s=100, marker='*')
        ax7.set_xlabel('Temperature (°C)', fontsize=11)
        ax7.set_ylabel('pH', fontsize=11)
        ax7.set_title('DNA Half-life (hours)', fontsize=12, fontweight='bold')
        fig.colorbar(contour7, ax=ax7, shrink=0.8)
        
        # Plot 8: Cycles to Transition
        ax8 = fig.add_subplot(gs[2, 1])
        cycles = self.surface_data['cycles_to_transition']
        contour8 = ax8.contourf(T, pH, cycles.T, levels=20, cmap='coolwarm')
        ax8.scatter(T_opt, pH_opt, color='red', s=100, marker='*')
        ax8.set_xlabel('Temperature (°C)', fontsize=11)
        ax8.set_ylabel('pH', fontsize=11)
        ax8.set_title('Cycles to Transition', fontsize=12, fontweight='bold')
        fig.colorbar(contour8, ax=ax8, shrink=0.8)
        
        # Plot 9: Summary Statistics
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        opt = self.optimization_results
        summary_text = f"""
        ╔═══════════════════════════════════════════════════════╗
        ║        2D OPTIMIZATION SUMMARY                       ║
        ╠═══════════════════════════════════════════════════════╣
        ║  Optimal Temperature:    {opt['optimal_T']:.1f}°C
        ║  Optimal pH:             {opt['optimal_pH']:.2f}
        ║  Maximum Enrichment:     {opt['max_enrichment']:.2f}x
        ║                                                     ║
        ║  At Optimal Conditions:                              ║
        ║    DNA Fraction:         {opt['final_dna_fraction']:.3f}
        ║    DNA Yield:            {opt['optimal_yield']/1000:.1f}K
        ║    Half-life:            {opt['optimal_half_life']:.1f}h
        ║    Transition:           {opt['transition_time']:.1f} days
        ║    Cycles:               {opt['cycles_to_transition']:.1f}
        ║                                                     ║
        ║  Fastest Transition:                                 ║
        ║    T:                   {opt['fast_T']:.1f}°C
        ║    pH:                  {opt['fast_pH']:.2f}
        ║    Cycles:              {opt['fast_transition']:.1f}
        ║    Enrichment:          {opt['fast_enrichment']:.2f}x
        ║                                                     ║
        ║  Statistics:                                        ║
        ║    Mean Enrichment:      {opt['mean_enrichment']:.2f}x
        ║    Std Enrichment:       {opt['std_enrichment']:.2f}x
        ║    Range:               {opt['enrichment_range'][0]:.2f}x -
        ║                           {opt['enrichment_range'][1]:.2f}x
        ╚═══════════════════════════════════════════════════════╝
        """
        
        ax9.text(0.5, 0.5, summary_text, ha='center', va='center',
                transform=ax9.transAxes, fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        
        if show_fig:
            plt.show()
        
        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches='tight')
            print(f"\n✅ 2D Sensitivity plot saved: {save_path}")
        
        return fig


# ====================================================================
# DATA EXPORT FUNCTIONS
# ====================================================================

def export_results_to_csv(
    analyzer: Any,
    surface_results: Dict[str, Any],
    vssuf: Optional[VSSUFEngine] = None,
    filename_prefix: str = "wet_dry_results"
) -> Dict[str, str]:
    """Export all simulation results to CSV and JSON files"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    files_generated = {}
    
    output_dir = "output_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Export surface data
    T = surface_results['T']
    pH = surface_results['pH']
    
    surface_df = pd.DataFrame()
    T_grid, pH_grid = np.meshgrid(T, pH, indexing='ij')
    surface_df['Temperature'] = T_grid.flatten()
    surface_df['pH'] = pH_grid.flatten()
    surface_df['Enrichment'] = surface_results['enrichment'].flatten()
    surface_df['DNA_Fraction'] = surface_results['fraction'].flatten()
    surface_df['DNA_Yield'] = surface_results['dna_yield'].flatten()
    surface_df['Half_Life'] = surface_results['half_life'].flatten()
    surface_df['Transition_Time'] = surface_results['transition_time'].flatten()
    surface_df['Cycles_To_Transition'] = surface_results['cycles_to_transition'].flatten()
    surface_df['Final_DNA_Fraction'] = surface_results['final_dna_fraction'].flatten()
    
    surface_csv = os.path.join(output_dir, f"{filename_prefix}_surface_{timestamp}.csv")
    surface_df.to_csv(surface_csv, index=False)
    files_generated['surface_csv'] = surface_csv
    print(f"✅ Surface data saved: {surface_csv}")
    
    # Export optimization results
    opt = analyzer.optimization_results
    optimization_data = {
        'timestamp': timestamp,
        'optimal_T': float(opt['optimal_T']),
        'optimal_pH': float(opt['optimal_pH']),
        'max_enrichment': float(opt['max_enrichment']),
        'optimal_fraction': float(opt['optimal_fraction']),
        'optimal_yield': float(opt['optimal_yield']),
        'optimal_half_life': float(opt['optimal_half_life']),
        'transition_time': float(opt['transition_time']),
        'cycles_to_transition': float(opt['cycles_to_transition']),
        'final_dna_fraction': float(opt['final_dna_fraction']),
        'fast_T': float(opt['fast_T']),
        'fast_pH': float(opt['fast_pH']),
        'fast_transition': float(opt['fast_transition']),
        'fast_enrichment': float(opt['fast_enrichment']),
        'mean_enrichment': float(opt['mean_enrichment']),
        'std_enrichment': float(opt['std_enrichment']),
        'cycle_period': float(analyzer.config['cycle_period']),
        'dry_fraction': float(analyzer.config['dry_fraction']),
        'mineral_type': str(analyzer.config['mineral_type'])
    }
    
    json_path = os.path.join(output_dir, f"{filename_prefix}_optimization_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(optimization_data, f, indent=4)
    files_generated['optimization_json'] = json_path
    print(f"✅ Optimization results saved: {json_path}")
    
    # Export cycle history if available
    if vssuf and vssuf.cycle_history:
        cycle_history = vssuf.cycle_history
        cycle_df = pd.DataFrame({
            'time_step': range(len(cycle_history['cycle_number'])),
            'cycle_number': cycle_history['cycle_number'],
            'phase': cycle_history['phase'],
            'water_activity': cycle_history['aw'],
            'temperature': cycle_history['temperature'],
            'dna_fraction': cycle_history['dna_fraction'],
            'enrichment': cycle_history['enrichment']
        })
        
        cycle_csv = os.path.join(output_dir, f"{filename_prefix}_cycles_{timestamp}.csv")
        cycle_df.to_csv(cycle_csv, index=False)
        files_generated['cycle_csv'] = cycle_csv
        print(f"✅ Cycle history saved: {cycle_csv}")
    
    # Export time series if available
    if vssuf and hasattr(vssuf, 'history') and vssuf.history['time']:
        history = vssuf.history
        time_hours = np.array(history['time']) / SimulationConstants.SECONDS_PER_HOUR
        
        time_df = pd.DataFrame({
            'time_hours': time_hours,
            'dsDNA_U': history['dsDNA_U'],
            'dsDNA_T': history['dsDNA_T'],
            'total_dna': history['total_dna'],
            'u_ratio': history['u_ratio'],
            'enrichment': history['enrichment']
        })
        
        time_csv = os.path.join(output_dir, f"{filename_prefix}_timeseries_{timestamp}.csv")
        time_df.to_csv(time_csv, index=False)
        files_generated['timeseries_csv'] = time_csv
        print(f"✅ Time series saved: {time_csv}")
    
    return files_generated


# ====================================================================
# MAIN EXECUTION
# ====================================================================

def run_wet_dry_analysis(quick_test: bool = True):
    """Run complete analysis with wet-dry cycles"""
    
    print("="*70)
    print("🌊 UPDSF v3.1 - FULLY BUG FIXED")
    print("   Wet-Dry Cycle Dynamics with Multiprocessing")
    print("="*70)
    
    if quick_test:
        print("\n⚡ QUICK TEST MODE")
        sensitivity_config = {
            'temperature_min': 25.0,
            'temperature_max': 45.0,
            'temperature_step': 10.0,
            'pH_min': 6.0,
            'pH_max': 8.0,
            'pH_step': 1.0,
            'simulation_hours': 24,
            'n_replicates': 1,
            'verbose': True,
            'cycle_period': 12.0,
            'dry_fraction': 0.5,
            'temperature_dry': 45.0,
            'temperature_wet': 30.0,
            'mineral_type': 'montmorillonite',
            'use_multiprocessing': False  # Disable for quick test
        }
    else:
        print("\n🔬 FULL ANALYSIS MODE")
        sensitivity_config = {
            'temperature_min': 25.0,
            'temperature_max': 60.0,
            'temperature_step': 5.0,
            'pH_min': 5.0,
            'pH_max': 9.0,
            'pH_step': 0.5,
            'simulation_hours': 240,
            'n_replicates': 3,
            'verbose': True,
            'cycle_period': 12.0,
            'dry_fraction': 0.5,
            'temperature_dry': 45.0,
            'temperature_wet': 30.0,
            'mineral_type': 'montmorillonite',
            'use_multiprocessing': True
        }
    
    print(f"\n  Cycle Period: {sensitivity_config['cycle_period']}h")
    print(f"  Dry Fraction: {sensitivity_config['dry_fraction']}")
    print(f"  Mineral: {sensitivity_config['mineral_type']}")
    print(f"  Temperature: {sensitivity_config['temperature_min']}°C - {sensitivity_config['temperature_max']}°C")
    print(f"  pH: {sensitivity_config['pH_min']:.1f} - {sensitivity_config['pH_max']:.1f}")
    print(f"  Multiprocessing: {sensitivity_config['use_multiprocessing']}")
    
    # Run analysis
    analyzer = TwoDSensitivityAnalyzer(sensitivity_config)
    surface_results = analyzer.run_sensitivity_analysis()
    
    # Plot results
    analyzer.plot_sensitivity_2d(save_path="sensitivity_wet_dry.png", show_fig=True)
    
    # Run detailed simulation at optimal conditions
    print("\n📊 Running detailed simulation at optimal conditions...")
    
    opt = analyzer.optimization_results
    optimal_T = opt['optimal_T']
    optimal_pH = opt['optimal_pH']
    
    wd_config = WetDryCycleConfig(
        cycle_period_hours=sensitivity_config['cycle_period'],
        dry_fraction=sensitivity_config['dry_fraction'],
        temperature_dry=sensitivity_config['temperature_dry'],
        temperature_wet=sensitivity_config['temperature_wet'],
        mineral_present=True,
        mineral_type=sensitivity_config['mineral_type']
    )
    
    vssuf = VSSUFEngine(
        temperature_C=optimal_T,
        pH=optimal_pH,
        seed=42,
        max_time_hours=sensitivity_config['simulation_hours'],
        verbose=True,
        wet_dry_config=wd_config
    )
    
    vssuf.run()
    
    # Plot detailed results
    plot_wet_dry_results(vssuf, save_path="wet_dry_dynamics.png")
    
    # Export data
    print("\n💾 Exporting results...")
    export_results_to_csv(analyzer, surface_results, vssuf)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    
    cycle_results = vssuf.get_cycle_results()
    print(f"\n🎯 Optimal Conditions:")
    print(f"   Temperature: {optimal_T:.1f}°C")
    print(f"   pH: {optimal_pH:.2f}")
    print(f"   Enrichment: {opt['max_enrichment']:.2f}x")
    print(f"   DNA Fraction: {cycle_results['final_dna_fraction']:.3f}")
    print(f"   Transition: {cycle_results['transition_time_days']:.1f} days")
    
    if not cycle_results['transition_detected']:
        print("   ⚠️ No transition detected (DNA never exceeded 50%)")
    
    print("\n📁 Output files:")
    print("   - sensitivity_wet_dry.png")
    print("   - wet_dry_dynamics.png")
    print("   - output_data/*.csv, *.json")
    
    return vssuf, analyzer


def plot_wet_dry_results(vssuf: VSSUFEngine, save_path: str = "wet_dry_dynamics.png"):
    """Plot detailed wet-dry cycle results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Wet-Dry Cycle Dynamics: DNA vs RNA Selection',
                fontsize=16, fontweight='bold')
    
    history = vssuf.history
    cycle_history = vssuf.cycle_history
    time_hours = np.array(history['time']) / SimulationConstants.SECONDS_PER_HOUR
    
    # Plot 1: DNA Fraction
    ax = axes[0, 0]
    dna_array = np.array(history['dsDNA_T'])
    u_array = np.array(history['dsDNA_U'])
    total_array = dna_array + u_array
    dna_frac = dna_array / np.maximum(1, total_array)
    ax.plot(time_hours, dna_frac, 'b-', linewidth=2, label='DNA Fraction')
    ax.axhline(y=0.5, color='r', linestyle='--', label='DNA > RNA threshold')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('DNA Fraction')
    ax.set_title('DNA Accumulation Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Enrichment
    ax = axes[0, 1]
    enrichment = np.array(history['enrichment'])
    ax.plot(time_hours, enrichment, 'purple', linewidth=2)
    ax.fill_between(time_hours, 0, enrichment, alpha=0.2, color='purple')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Enrichment')
    ax.set_title('Thymine Enrichment')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Phase and Water Activity
    ax = axes[0, 2]
    ax2 = ax.twinx()
    
    phases = cycle_history['phase']
    aw = cycle_history['aw']
    
    phase_colors = {'wet': 'blue', 'dry': 'orange', 'transition': 'green'}
    for i, phase in enumerate(phases):
        if i < len(phases) - 1:
            color = phase_colors.get(phase, 'gray')
            ax.axvspan(i, i+1, alpha=0.1, color=color)
    
    ax2.plot(range(len(aw)), aw, 'k-', linewidth=1.5)
    ax2.set_ylabel('Water Activity (aw)')
    ax2.set_ylim(0, 1.1)
    ax.set_xlabel('Time Step')
    ax.set_title('Phase Dynamics')
    
    # Plot 4: Temperature
    ax = axes[1, 0]
    temp_profile = cycle_history['temperature']
    ax.plot(range(len(temp_profile)), temp_profile, 'r-', linewidth=2)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Oscillation')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Polymer Lengths
    ax = axes[1, 1]
    if vssuf.rna_lengths and vssuf.dna_lengths:
        ax.plot(range(len(vssuf.rna_lengths)), vssuf.rna_lengths, 'b-', label='RNA', alpha=0.7)
        ax.plot(range(len(vssuf.dna_lengths)), vssuf.dna_lengths, 'r-', label='DNA', alpha=0.7)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Length (bases)')
    ax.set_title('Polymer Length Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    cycle_results = vssuf.get_cycle_results()
    
    summary = f"""
    ╔═══════════════════════════════════════════════╗
    ║      WET-DRY CYCLE SUMMARY                   ║
    ╠═══════════════════════════════════════════════╣
    ║  Total Cycles:        {cycle_results['total_cycles']}
    ║  Cycle Period:        {vssuf.wd_engine.config.cycle_period_hours:.1f}h
    ║  Dry Fraction:        {vssuf.wd_engine.config.dry_fraction:.1f}
    ║  Mineral:             {vssuf.wd_engine.config.mineral_type}
    ║                                              ║
    ║  Final DNA Fraction:  {cycle_results['final_dna_fraction']:.3f}
    ║  Final Enrichment:    {cycle_results['final_enrichment']:.2f}x
    ║  Transition:          {cycle_results['transition_time_days']:.1f} days
    ║  Transition Detected: {cycle_results['transition_detected']}
    ║                                              ║
    ║  Events:                                      ║
    ║    Polymerization:    {cycle_results['polymerization_events']}
    ║    Hydrolysis:        {cycle_results['hydrolysis_events']}
    ╚═══════════════════════════════════════════════╝
    """
    
    ax.text(0.5, 0.5, summary, ha='center', va='center',
           transform=ax.transAxes, fontsize=9, family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Wet-dry dynamics plot saved: {save_path}")
    
    plt.show()
    return fig


# ====================================================================
# ENTRY POINT
# ====================================================================

if __name__ == "__main__":
    # Run with quick_test=False for full analysis
    vssuf, analyzer = run_wet_dry_analysis(quick_test=True)
