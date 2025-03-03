###################################################################################################
#   cicada_histos_script.py                                                                       #
#   Description: process cicada-triggered events and save relevant observables in histograms      #
#   Authors: Elliott Kauffman                                                                     #
# ##################################################################################################

###################################################################################################
# IMPORTS

# library imports
import awkward as ak
from collections import defaultdict
import dask
from dask.distributed import Client
import dask_awkward as dak
import dill
import hist
import hist.dask as hda
import json
import numpy as np
import time
import vector
vector.register_awkward()

# coffea imports
from coffea.nanoevents.methods import vector
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import coffea.processor as processor
from coffea.dataset_tools import (
    apply_to_fileset,
    max_chunks,
    preprocess,
)

NanoAODSchema.warn_missing_crossrefs = False

###################################################################################################
# PROCESSING OPTIONS

json_filename = "2024_data_reco.json"
dataset_name = "Reco_2024I"
golden_json_path = "2024I_Golden.json"

n_files = -1                                                 # number of files to process (-1 for all)
coffea_step_size = 5_000                                    # step size for coffea processor
coffea_files_per_batch = 1                                   # files per batch for coffea processor
files_per_chunk = 10                                        # number of files to process at a time
visualize = False


# which hists to save (comment out unwanted)
hist_selection = [
    # "lead_recojet_pt",
    # "lead_recojet_eta",
    # "lead_recojet_phi",
    "lead_recojet_eta_phi",
    "lead_recojet_ieta_iphi",
    # "lead_recoelectron_pt",
    # "lead_recoelectron_eta",
    # "lead_recoelectron_phi",
    "lead_recoelectron_eta_phi",
    "lead_recoelectron_ieta_iphi",
    # "lead_recophoton_pt",
    # "lead_recophoton_eta",
    # "lead_recophoton_phi",
    "lead_recophoton_eta_phi",
    "lead_recophoton_ieta_iphi",
    # "lead_l1jet_pt",
    # "lead_l1jet_eta",
    # "lead_l1jet_phi",
    "lead_l1jet_eta_phi",
    "lead_l1jet_ieta_iphi",
    # "lead_l1eg_pt",
    # "lead_l1eg_eta",
    # "lead_l1eg_phi",
    "lead_l1eg_eta_phi",
    "lead_l1eg_ieta_iphi",
]

# which triggers to save (comment out unwanted or add)
triggers = [
    'L1_CICADA_Medium', 
    'L1_ZeroBias', 
    'L1_AXO_Nominal'
]

###################################################################################################
# DEFINE SCHEMA
class ScoutingNanoAODSchema(NanoAODSchema):
    """ScoutingNano schema builder

    ScoutingNano is a NanoAOD format that includes Scouting objects
    """

    mixins = {
        **NanoAODSchema.mixins,
    }
    all_cross_references = {
        **NanoAODSchema.all_cross_references
    }

###################################################################################################
# HELPER FUNCTIONS FOR PROCESSOR

def createHist_3d(
    hist_dict, dataset_axis, trigger_axis, observable1_axis, observable2_axis, observable3_axis, hist_name 
):
    h = hda.hist.Hist(
        dataset_axis, 
        trigger_axis, 
        observable1_axis, 
        observable2_axis, 
        observable3_axis,
        storage="weight", 
        label="nEvents")
    hist_dict[f'{hist_name}'] = h
    
    return hist_dict

def createHist_2d(
    hist_dict, dataset_axis, trigger_axis, observable1_axis, observable2_axis, hist_name 
):
    h = hda.hist.Hist(
        dataset_axis, 
        trigger_axis, 
        observable1_axis, 
        observable2_axis, 
        storage="weight", 
        label="nEvents")
    hist_dict[f'{hist_name}'] = h
    
    return hist_dict

def createHist_1d(
    hist_dict, dataset_axis, trigger_axis, observable_axis, hist_name 
):
    h = hda.hist.Hist(
        dataset_axis, 
        trigger_axis, 
        observable_axis, 
        storage="weight", 
        label="nEvents")
    hist_dict[f'{hist_name}'] = h
    
    return hist_dict

def fillHist_3d(
    hist_dict, hist_name, dataset, observable1, observable2, observable3, observable1_name, observable2_name, observable3_name, trigger_path
):
    
    kwargs = {
        observable1_name: observable1,
        observable2_name: observable2,
        observable3_name: observable3,
        "dataset": dataset,
        "trigger": trigger_path,
    }
    
    hist_dict[f'{hist_name}'].fill(**kwargs)
    
    return hist_dict

def fillHist_2d(
    hist_dict, hist_name, dataset, observable1, observable2, observable1_name, observable2_name, trigger_path
):
    
    kwargs = {
        observable1_name: observable1,
        observable2_name: observable2,
        "dataset": dataset,
        "trigger": trigger_path,
    }
    
    hist_dict[f'{hist_name}'].fill(**kwargs)
    
    return hist_dict

def fillHist_1d(
    hist_dict, hist_name, dataset, observable, observable_name, trigger_path
):
    
    kwargs = {
        observable_name: observable,
        "dataset": dataset,
        "trigger": trigger_path,
    }
    
    hist_dict[f'{hist_name}'].fill(**kwargs)
    
    return hist_dict


###################################################################################################
# DEFINE COFFEA PROCESSOR
class MakeHists (processor.ProcessorABC):
    def __init__(
        self, 
        trigger_paths=[],
        hists_to_process=[],
        lumi_mask=None,
    ):
        self.trigger_paths = trigger_paths
        self.hists_to_process = hists_to_process
        self.lumi_mask = lumi_mask
        
        # define axes for histograms
        self.dataset_axis = hist.axis.StrCategory(
            [], growth=True, name="dataset", label="Primary dataset"
        )
        self.trigger_axis = hist.axis.StrCategory(
            [], growth=True, name="trigger", label="Trigger"
        )
        self.pt_axis = hist.axis.Regular(
            500, 0, 5000, name="pt", label=r"$p_{T}$ [GeV]"
        )
        self.pt_max_axis = hist.axis.IntCategory(
            [], growth=True, name="pt_min", label=r"$p_{T}$ [GeV]"
        )
        self.pt_min_axis = hist.axis.IntCategory(
            [], growth=True, name="pt_max", label=r"$p_{T}$ [GeV]"
        )
        self.eta_axis = hist.axis.Regular(
            150, -5, 5, name="eta", label=r"$\eta$"
        )
        self.phi_axis = hist.axis.Regular(
            30, -4, 4, name="phi", label=r"$\phi$"
        )
        
        ieta_bins = [-5.0,-4.5,-4.0,-3.5,-3.0,-2.172,-1.74,-1.392,
                     -1.044,-0.696,-0.348,0.0,0.348,0.696,1.044,
                     1.392,1.74,2.172,3.0,3.5,4.0,4.5,5.0]
        self.ieta_axis = hist.axis.Variable(
            ieta_bins, name="ieta", label=r"i$\eta$"
        )
        
        iphi_bins = [-2.967,-2.618,-2.269,-1.919,-1.57,-1.221,
                     -0.8726,-0.5235,-0.1745,0.1745,0.5235,0.8726,
                     1.221,1.57,1.919,2.269,2.618,2.967,3.316]
        self.iphi_axis = hist.axis.Variable(
            iphi_bins, name="iphi", label=r"i$\phi$"
        )
        
    def is_valid_lumi(self, runs, lumis):
        """
        Lumi mask filtering
        """
        mask = ak.zeros_like(runs, dtype=bool)
        for run, lumisections in self.lumi_mask.items():
            in_run = runs == int(run)
            for lumi_min, lumi_max in lumisections:
                in_lumi = (lumis >= int(lumi_min)) & (lumis <= int(lumi_max))
                mask = mask | (in_run & in_lumi)
        return mask
        
    def process(self, events):
        dataset = events.metadata['dataset']
        hist_dict = {}
        return_dict = {}
        
        valid_lumis = self.is_valid_lumi(events.run, events.luminosityBlock)
        events = events[valid_lumis]
               
        # Saturated-Jets event cut
        events = events[dak.all(events.L1Jet.pt<1000,axis=1)]
        # Saturated-MET event cut
        events = events[dak.flatten(events.L1EtSum.pt[(events.L1EtSum.etSumType==2) 
                                                      & (events.L1EtSum.bx==0)])<1040]
        
        if ("lead_recojet_pt" in self.hists_to_process):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.pt_axis, 
                "lead_recojet_pt"
            )
        if ("lead_recojet_eta" in self.hists_to_process):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.eta_axis, 
                self.pt_axis,
                "lead_recojet_eta" 
            )
        if ("lead_recojet_phi" in self.hists_to_process):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.phi_axis, 
                self.pt_axis,
                "lead_recojet_phi" 
            )
        if ("lead_recojet_eta_phi" in self.hists_to_process):
            hist_dict = createHist_3d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.eta_axis, 
                self.phi_axis,
                self.pt_axis,
                "lead_recojet_eta_phi" 
            )
        if ("lead_recojet_ieta_iphi" in self.hists_to_process):
            hist_dict = createHist_3d(
                hist_dict, 
                self.dataset_axis,
                self.trigger_axis, 
                self.ieta_axis, 
                self.iphi_axis, 
                self.pt_axis,
                "lead_recojet_ieta_iphi" 
            )
        if ("lead_recoelectron_pt" in self.hists_to_process):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.pt_axis, 
                "lead_recoelectron_pt"
            )
        if ("lead_recoelectron_eta" in self.hists_to_process):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.eta_axis, 
                self.pt_axis,
                "lead_recoelectron_eta" 
            )
        if ("lead_recoelectron_phi" in self.hists_to_process):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.phi_axis, 
                self.pt_axis,
                "lead_recoelectron_phi" 
            )
        if ("lead_recoelectron_eta_phi" in self.hists_to_process):
            hist_dict = createHist_3d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.eta_axis, 
                self.phi_axis,
                self.pt_axis,
                "lead_recoelectron_eta_phi" 
            )
        if ("lead_recoelectron_ieta_iphi" in self.hists_to_process):
            hist_dict = createHist_3d(
                hist_dict, 
                self.dataset_axis,
                self.trigger_axis, 
                self.ieta_axis, 
                self.iphi_axis, 
                self.pt_axis,
                "lead_recoelectron_ieta_iphi" 
            )
        if ("lead_recophoton_pt" in self.hists_to_process):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.pt_axis, 
                "lead_recophoton_pt"
            )
        if ("lead_recophoton_eta" in self.hists_to_process):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.eta_axis, 
                self.pt_axis,
                "lead_recophoton_eta" 
            )
        if ("lead_recophoton_phi" in self.hists_to_process):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.phi_axis, 
                self.pt_axis,
                "lead_recophoton_phi" 
            )
        if ("lead_recophoton_eta_phi" in self.hists_to_process):
            hist_dict = createHist_3d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.eta_axis, 
                self.phi_axis,
                self.pt_axis,
                "lead_recophoton_eta_phi" 
            )
        if ("lead_recophoton_ieta_iphi" in self.hists_to_process):
            hist_dict = createHist_3d(
                hist_dict, 
                self.dataset_axis,
                self.trigger_axis, 
                self.ieta_axis, 
                self.iphi_axis, 
                self.pt_axis,
                "lead_recophoton_ieta_iphi" 
            )
        if ("lead_l1jet_pt" in self.hists_to_process):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.pt_axis, 
                "lead_l1jet_pt"
            )
        if ("lead_l1jet_eta" in self.hists_to_process):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.eta_axis, 
                self.pt_axis,
                "lead_l1jet_eta" 
            )
        if ("lead_l1jet_phi" in self.hists_to_process):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.phi_axis, 
                self.pt_axis,
                "lead_l1jet_phi" 
            )
        if ("lead_l1jet_eta_phi" in self.hists_to_process):
            hist_dict = createHist_3d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.eta_axis, 
                self.phi_axis, 
                self.pt_axis,
                "lead_l1jet_eta_phi" 
            )
        if ("lead_l1jet_ieta_iphi" in self.hists_to_process):
            hist_dict = createHist_3d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.ieta_axis, 
                self.iphi_axis, 
                self.pt_axis,
                "lead_l1jet_ieta_iphi" 
            )
        if ("lead_l1eg_pt" in self.hists_to_process):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.pt_axis, 
                "lead_l1eg_pt"
            )
        if ("lead_l1eg_eta" in self.hists_to_process):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.eta_axis, 
                self.pt_axis,
                "lead_l1eg_eta" 
            )
        if ("lead_l1eg_phi" in self.hists_to_process):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.phi_axis, 
                self.pt_axis,
                "lead_l1eg_phi" 
            )
        if ("lead_l1eg_eta_phi" in self.hists_to_process):
            hist_dict = createHist_3d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.eta_axis, 
                self.phi_axis, 
                self.pt_axis,
                "lead_l1eg_eta_phi" 
            )
        if ("lead_l1eg_ieta_iphi" in self.hists_to_process):
            hist_dict = createHist_3d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.ieta_axis, 
                self.iphi_axis, 
                self.pt_axis,
                "lead_l1eg_ieta_iphi" 
            )
            
        # loop over trigger paths
        for trigger_path in self.trigger_paths:
            trig_br = getattr(events,trigger_path.split('_')[0])
            trig_path = '_'.join(trigger_path.split('_')[1:])
            events_trig = events[getattr(trig_br,trig_path)]
            
            recojet_mask = ( 
                #(abs(events_trig.Jet.eta) < 2.4) &      # remove forward jets
                (events_trig.Jet.neHEF < 0.99) &        # neutral hadron fraction
                (events_trig.Jet.neEmEF < 0.90) &       # neutral em fraction
                (events_trig.Jet.nConstituents > 1) &   # number of constituents
                (events_trig.Jet.muEF < 0.80) &         # muon fraction
                (events_trig.Jet.chHEF > 0.01) &        # charged hadron fraction
                (events_trig.Jet.chMultiplicity > 0) &             # charged multiplicity
                (events_trig.Jet.chEmEF < 0.80)         # charged em fraction
            )
            
            selected_recojets_trig = events_trig.Jet[recojet_mask]
                
            # require at least one valid jet per event
            selected_recojets_trig = selected_recojets_trig[ak.num(selected_recojets_trig,axis=1)>=1]
            selected_recojets_trig.phi_mod = dak.where(
                selected_recojets_trig.phi < -2.967, 
                selected_recojets_trig.phi  + 2 * np.pi, 
                selected_recojets_trig.phi
            )
                                        
            if ("lead_recojet_pt" in self.hists_to_process):
                lead_recojet_pt = selected_recojets_trig.pt[:, 0]
                hist_dict = fillHist_1d(
                    hist_dict, 
                    "lead_recojet_pt", 
                    dataset, 
                    lead_recojet_pt, 
                    "pt", 
                    trigger_path
                )
            if ("lead_recojet_eta" in self.hists_to_process):
                lead_recojet_eta = selected_recojets_trig.eta[:, 0]
                lead_recojet_pt = selected_recojets_trig.pt[:, 0]
                hist_dict = fillHist_2d(
                    hist_dict, 
                    'lead_recojet_eta', 
                    dataset, 
                    lead_recojet_eta, 
                    lead_recojet_pt,
                    "eta",
                    "pt",
                    trigger_path, 
                )
            if ("lead_recojet_phi" in self.hists_to_process):
                lead_recojet_phi = selected_recojets_trig.phi[:, 0]
                lead_recojet_pt = selected_recojets_trig.pt[:, 0]
                hist_dict = fillHist_2d(
                    hist_dict, 
                    'lead_recojet_phi', 
                    dataset, 
                    lead_recojet_phi, 
                    lead_recojet_pt,
                    "phi",
                    "pt",
                    trigger_path, 
                )
            if ("lead_recojet_eta_phi" in self.hists_to_process):
                lead_recojet_eta = selected_recojets_trig.eta[:, 0]
                lead_recojet_phi = selected_recojets_trig.phi[:, 0]
                lead_recojet_pt = selected_recojets_trig.pt[:, 0]
                hist_dict = fillHist_3d(
                    hist_dict, 
                    'lead_recojet_eta_phi', 
                    dataset, 
                    lead_recojet_eta, 
                    lead_recojet_phi,
                    lead_recojet_pt,
                    "eta",
                    "phi",
                    "pt",
                    trigger_path, 
                )
            if ("lead_recojet_ieta_iphi" in self.hists_to_process):
                lead_recojet_eta = selected_recojets_trig.eta[:, 0]
                lead_recojet_phi = selected_recojets_trig.phi_mod[:, 0]
                lead_recojet_pt = selected_recojets_trig.pt[:, 0]
                hist_dict = fillHist_3d(
                    hist_dict, 
                    'lead_recojet_ieta_iphi', 
                    dataset, 
                    lead_recojet_eta, 
                    lead_recojet_phi,
                    lead_recojet_pt,
                    "ieta",
                    "iphi",
                    "pt",
                    trigger_path, 
                )
                
            selected_recoelectrons_trig = events_trig.Electron[ak.num(events_trig.Electron,axis=1)>=1]
            selected_recoelectrons_trig.phi_mod = dak.where(
                selected_recoelectrons_trig.phi < -2.967, 
                selected_recoelectrons_trig.phi  + 2 * np.pi, 
                selected_recoelectrons_trig.phi
            )
                                        
            if ("lead_recoelectron_pt" in self.hists_to_process):
                lead_recoelectron_pt = selected_recoelectrons_trig.pt[:, 0]
                hist_dict = fillHist_1d(
                    hist_dict, 
                    "lead_recoelectron_pt", 
                    dataset, 
                    lead_recoelectron_pt, 
                    "pt", 
                    trigger_path
                )
            if ("lead_recoelectron_eta" in self.hists_to_process):
                lead_recoelectron_eta = selected_recoelectrons_trig.eta[:, 0]
                lead_recoelectron_pt = selected_recoelectrons_trig.pt[:, 0]
                hist_dict = fillHist_2d(
                    hist_dict, 
                    'lead_recoelectron_eta', 
                    dataset, 
                    lead_recoelectron_eta, 
                    lead_recoelectron_pt,
                    "eta",
                    "pt",
                    trigger_path, 
                )
            if ("lead_recoelectron_phi" in self.hists_to_process):
                lead_recoelectron_phi = selected_recoelectrons_trig.phi[:, 0]
                lead_recoelectron_pt = selected_recoelectrons_trig.pt[:, 0]
                hist_dict = fillHist_2d(
                    hist_dict, 
                    'lead_recoelectron_phi', 
                    dataset, 
                    lead_recoelectron_phi, 
                    lead_recoelectron_pt,
                    "phi",
                    "pt",
                    trigger_path, 
                )
            if ("lead_recoelectron_eta_phi" in self.hists_to_process):
                lead_recoelectron_eta = selected_recoelectrons_trig.eta[:, 0]
                lead_recoelectron_phi = selected_recoelectrons_trig.phi[:, 0]
                lead_recoelectron_pt = selected_recoelectrons_trig.pt[:, 0]
                hist_dict = fillHist_3d(
                    hist_dict, 
                    'lead_recoelectron_eta_phi', 
                    dataset, 
                    lead_recoelectron_eta, 
                    lead_recoelectron_phi,
                    lead_recoelectron_pt,
                    "eta",
                    "phi",
                    "pt",
                    trigger_path, 
                )
            if ("lead_recoelectron_ieta_iphi" in self.hists_to_process):
                lead_recoelectron_eta = selected_recoelectrons_trig.eta[:, 0]
                lead_recoelectron_phi = selected_recoelectrons_trig.phi_mod[:, 0]
                lead_recoelectron_pt = selected_recoelectrons_trig.pt[:, 0]
                hist_dict = fillHist_3d(
                    hist_dict, 
                    'lead_recoelectron_ieta_iphi', 
                    dataset, 
                    lead_recoelectron_eta, 
                    lead_recoelectron_phi,
                    lead_recoelectron_pt,
                    "ieta",
                    "iphi",
                    "pt",
                    trigger_path, 
                )
                
            selected_recophotons_trig = events_trig.Photon[ak.num(events_trig.Photon,axis=1)>=1]
            selected_recophotons_trig.phi_mod = dak.where(
                selected_recophotons_trig.phi < -2.967, 
                selected_recophotons_trig.phi  + 2 * np.pi, 
                selected_recophotons_trig.phi
            )
                                        
            if ("lead_recophoton_pt" in self.hists_to_process):
                lead_recophoton_pt = selected_recophotons_trig.pt[:, 0]
                hist_dict = fillHist_1d(
                    hist_dict, 
                    "lead_recophoton_pt", 
                    dataset, 
                    lead_recophoton_pt, 
                    "pt", 
                    trigger_path
                )
            if ("lead_recophoton_eta" in self.hists_to_process):
                lead_recophoton_eta = selected_recophotons_trig.eta[:, 0]
                lead_recophoton_pt = selected_recophotons_trig.pt[:, 0]
                hist_dict = fillHist_2d(
                    hist_dict, 
                    'lead_recophoton_eta', 
                    dataset, 
                    lead_recophoton_eta, 
                    lead_recophoton_pt,
                    "eta",
                    "pt",
                    trigger_path, 
                )
            if ("lead_recophoton_phi" in self.hists_to_process):
                lead_recophoton_phi = selected_recophotons_trig.phi[:, 0]
                lead_recophoton_pt = selected_recophotons_trig.pt[:, 0]
                hist_dict = fillHist_2d(
                    hist_dict, 
                    'lead_recophoton_phi', 
                    dataset, 
                    lead_recophoton_phi, 
                    lead_recophoton_pt,
                    "phi",
                    "pt",
                    trigger_path, 
                )
            if ("lead_recophoton_eta_phi" in self.hists_to_process):
                lead_recophoton_eta = selected_recophotons_trig.eta[:, 0]
                lead_recophoton_phi = selected_recophotons_trig.phi[:, 0]
                lead_recophoton_pt = selected_recophotons_trig.pt[:, 0]
                hist_dict = fillHist_3d(
                    hist_dict, 
                    'lead_recophoton_eta_phi', 
                    dataset, 
                    lead_recophoton_eta, 
                    lead_recophoton_phi,
                    lead_recophoton_pt,
                    "eta",
                    "phi",
                    "pt",
                    trigger_path, 
                )
            if ("lead_recophoton_ieta_iphi" in self.hists_to_process):
                lead_recophoton_eta = selected_recophotons_trig.eta[:, 0]
                lead_recophoton_phi = selected_recophotons_trig.phi_mod[:, 0]
                lead_recophoton_pt = selected_recophotons_trig.pt[:, 0]
                hist_dict = fillHist_3d(
                    hist_dict, 
                    'lead_recophoton_ieta_iphi', 
                    dataset, 
                    lead_recophoton_eta, 
                    lead_recophoton_phi,
                    lead_recophoton_pt,
                    "ieta",
                    "iphi",
                    "pt",
                    trigger_path, 
                )
                
            selected_l1jets_trig = events_trig.L1Jet[ak.num(events_trig.L1Jet,axis=1)>=1]
            selected_l1jets_trig.phi_mod = dak.where(
                selected_l1jets_trig.phi < -2.967, 
                selected_l1jets_trig.phi  + 2 * np.pi, 
                selected_l1jets_trig.phi
            )
            
            if ("lead_l1jet_pt" in self.hists_to_process):
                lead_l1jet_pt = selected_l1jets_trig.pt[:, 0]
                hist_dict = fillHist_1d(
                    hist_dict, 
                    "lead_l1jet_pt", 
                    dataset, 
                    lead_l1jet_pt, 
                    "pt", 
                    trigger_path
                )
            if ("lead_l1jet_eta" in self.hists_to_process):
                lead_l1jet_eta = selected_l1jets_trig.eta[:, 0]
                lead_l1jet_pt = selected_l1jets_trig.pt[:, 0]
                hist_dict = fillHist_2d(
                    hist_dict, 
                    'lead_l1jet_eta', 
                    dataset, 
                    lead_l1jet_eta, 
                    lead_l1jet_pt,
                    "eta",
                    "pt",
                    trigger_path, 
                )
            if ("lead_l1jet_phi" in self.hists_to_process):
                lead_l1jet_phi = selected_l1jets_trig.phi[:, 0]
                lead_l1jet_pt = selected_l1jets_trig.pt[:, 0]
                hist_dict = fillHist_2d(
                    hist_dict, 
                    'lead_l1jet_phi', 
                    dataset, 
                    lead_l1jet_phi, 
                    lead_l1jet_pt,
                    "phi",
                    "pt",
                    trigger_path, 
                )
            if ("lead_l1jet_eta_phi" in self.hists_to_process):
                lead_l1jet_eta = selected_l1jets_trig.eta[:, 0]
                lead_l1jet_phi = selected_l1jets_trig.phi[:, 0]
                lead_l1jet_pt = selected_l1jets_trig.pt[:, 0]
                hist_dict = fillHist_3d(
                    hist_dict, 
                    'lead_l1jet_eta_phi', 
                    dataset, 
                    lead_l1jet_eta, 
                    lead_l1jet_phi,
                    lead_l1jet_pt,
                    "eta",
                    "phi",
                    "pt",
                    trigger_path, 
                )
            if ("lead_l1jet_ieta_iphi" in self.hists_to_process):
                lead_l1jet_eta = selected_l1jets_trig.eta[:, 0]
                lead_l1jet_phi = selected_l1jets_trig.phi_mod[:, 0]
                lead_l1jet_pt = selected_l1jets_trig.pt[:, 0]
                hist_dict = fillHist_3d(
                    hist_dict, 
                    'lead_l1jet_ieta_iphi', 
                    dataset, 
                    lead_l1jet_eta, 
                    lead_l1jet_phi,
                    lead_l1jet_pt,
                    "ieta",
                    "iphi",
                    "pt",
                    trigger_path, 
                )
                
            selected_l1egs_trig = events_trig.L1EG[ak.num(events_trig.L1EG,axis=1)>=1]
            selected_l1egs_trig.phi_mod = dak.where(
                selected_l1egs_trig.phi < -2.967, 
                selected_l1egs_trig.phi  + 2 * np.pi, 
                selected_l1egs_trig.phi
            )
            
            if ("lead_l1eg_pt" in self.hists_to_process):
                lead_l1eg_pt = selected_l1egs_trig.pt[:, 0]
                hist_dict = fillHist_1d(
                    hist_dict, 
                    "lead_l1eg_pt", 
                    dataset, 
                    lead_l1eg_pt, 
                    "pt", 
                    trigger_path
                )
            if ("lead_l1eg_eta" in self.hists_to_process):
                lead_l1eg_eta = selected_l1egs_trig.eta[:, 0]
                lead_l1eg_pt = selected_l1egs_trig.pt[:, 0]
                hist_dict = fillHist_2d(
                    hist_dict, 
                    'lead_l1eg_eta', 
                    dataset, 
                    lead_l1eg_eta, 
                    lead_l1eg_pt,
                    "eta",
                    "pt",
                    trigger_path, 
                )
            if ("lead_l1eg_phi" in self.hists_to_process):
                lead_l1eg_phi = selected_l1egs_trig.phi[:, 0]
                lead_l1eg_pt = selected_l1egs_trig.pt[:, 0]
                hist_dict = fillHist_2d(
                    hist_dict, 
                    'lead_l1eg_phi', 
                    dataset, 
                    lead_l1eg_phi, 
                    lead_l1eg_pt,
                    "phi",
                    "pt",
                    trigger_path, 
                )
            if ("lead_l1eg_eta_phi" in self.hists_to_process):
                lead_l1eg_eta = selected_l1egs_trig.eta[:, 0]
                lead_l1eg_phi = selected_l1egs_trig.phi[:, 0]
                lead_l1eg_pt = selected_l1egs_trig.pt[:, 0]
                hist_dict = fillHist_3d(
                    hist_dict, 
                    'lead_l1jet_eta_phi', 
                    dataset, 
                    lead_l1eg_eta, 
                    lead_l1eg_phi,
                    lead_l1eg_pt,
                    "eta",
                    "phi",
                    "pt",
                    trigger_path, 
                )
            if ("lead_l1eg_ieta_iphi" in self.hists_to_process):
                lead_l1eg_eta = selected_l1egs_trig.eta[:, 0]
                lead_l1eg_phi = selected_l1egs_trig.phi_mod[:, 0]
                lead_l1eg_pt = selected_l1egs_trig.pt[:, 0]
                hist_dict = fillHist_3d(
                    hist_dict, 
                    'lead_l1eg_ieta_iphi', 
                    dataset, 
                    lead_l1eg_eta, 
                    lead_l1eg_phi,
                    lead_l1eg_pt,
                    "ieta",
                    "iphi",
                    "pt",
                    trigger_path, 
                )
                    
        return_dict['hists'] = hist_dict
        return_dict['trigger'] = self.trigger_paths if len(self.trigger_paths)>0 else None
                
        return return_dict

    def postprocess(self, accumulator):
        return accumulator


###################################################################################################
# DEFINE MAIN FUNCTION
def main():
    client = Client("tls://localhost:8786")
    
    with open(golden_json_path) as json_file:
        golden_json = json.load(json_file)
    
    with open(json_filename) as json_file:
        dataset = json.load(json_file)
    
    dataset_skimmed = {dataset_name: {'files': {}}}
    i = 0
    for key, value in dataset[dataset_name]['files'].items():
        if ((i<n_files) or (n_files==-1)):
            dataset_skimmed[dataset_name]['files'][key] = value
        i+=1
         
    number_of_files = n_files
    if n_files==-1: number_of_files = i
    print(f"Processing {number_of_files} files")
    
    # calculate chunks
    if number_of_files > files_per_chunk:
        chunks_left = np.arange(0,number_of_files,files_per_chunk,dtype=int)
        print(chunks_left)
        chunks_right = np.arange(files_per_chunk-1,number_of_files,files_per_chunk,dtype=int)
        chunks_right = np.append(chunks_right, number_of_files)
        print(chunks_right)
    else:
        chunks_left = [0]
        chunks_right = [number_of_files]
        
    print("Number of chunks = ", len(chunks_left))
    
    # iterate over chunks and run coffea processor
    for j in range(10,len(chunks_left)):
        print("Current chunk = ", j)
        
        dataset_reskimmed = {dataset_name: {'files': {}}}
        i = 0
        for key, value in dataset_skimmed[dataset_name]['files'].items():
            if (i>=chunks_left[j]) and (i<=chunks_right[j]):
                dataset_reskimmed[dataset_name]['files'][key] = value
            i+=1
        
        print("Number of Files to Process This Chunk = ", len(dataset_reskimmed[dataset_name]['files']))
        
        dataset_runnable, dataset_updated = preprocess(
            dataset_reskimmed,
            align_clusters=False,
            step_size=coffea_step_size,
            files_per_batch=coffea_files_per_batch,
            skip_bad_files=True,
            save_form=False,
        )

        tstart = time.time()
    
        to_compute = apply_to_fileset(
            MakeHists(trigger_paths=triggers, 
                      hists_to_process=hist_selection,
                      lumi_mask = golden_json,
                     ),
            max_chunks(dataset_runnable, 300000),
            schemaclass=ScoutingNanoAODSchema,
            uproot_options={"allow_read_errors_with_report": (OSError, TypeError, KeyError)}
        )
    
        if visualize:
            dask.optimize(to_compute)
            dask.visualize(to_compute, filename="dask_coffea_graph_ttbar", format="pdf")
        
        (hist_result,) = dask.compute(to_compute)
        print(f'Chunk took {time.time()-tstart:.1f}s to process')
        hist_result = hist_result[0]

        #Save file 
        with open(f'hist_result_{dataset_name}_ttbar_chunk{j}.pkl', 'wb') as file:
                # dump information to that file
                dill.dump(hist_result, file)
    

###################################################################################################
# RUN SCRIPT
if __name__=="__main__":
    main()
