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

json_filename = "2024_data_scouting.json"
dataset_name = "Scouting_2024I"

n_files = -1                                                 # number of files to process (-1 for all)
coffea_step_size = 15_000                                    # step size for coffea processor
coffea_files_per_batch = 1                                   # files per batch for coffea processor
files_per_chunk = 10                                        # number of files to process at a time
visualize = False

# which hists to save (comment out unwanted)
hist_selection = [
    # "lead_scoutingjet_pt",
    # "lead_scoutingjet_eta",
    # "lead_scoutingjet_phi",
    "lead_scoutingjet_eta_phi",
    "lead_scoutingjet_ieta_iphi",
    # "lead_scoutingelectron_pt",
    # "lead_scoutingelectron_eta",
    # "lead_scoutingelectron_phi",
    "lead_scoutingelectron_eta_phi",
    "lead_scoutingelectron_ieta_iphi",
    # "lead_scoutingphoton_pt",
    # "lead_scoutingphoton_eta",
    # "lead_scoutingphoton_phi",
    "lead_scoutingphoton_eta_phi",
    "lead_scoutingphoton_ieta_iphi",
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
    # 'DST_PFScouting_AXOLoose', 
    'DST_PFScouting_AXONominal', 
    # 'DST_PFScouting_AXOTight', 
    # 'DST_PFScouting_AXOVLoose', 
    # 'DST_PFScouting_AXOVTight',
    # 'DST_PFScouting_CICADALoose', 
    'DST_PFScouting_CICADAMedium', 
    # 'DST_PFScouting_CICADATight', 
    # 'DST_PFScouting_CICADAVLoose', 
    # 'DST_PFScouting_CICADAVTight',
    # 'DST_PFScouting_DoubleMuon',
    'DST_PFScouting_JetHT',
    'DST_PFScouting_ZeroBias'
]

###################################################################################################
# DEFINE SCHEMA
class ScoutingNanoAODSchema(NanoAODSchema):
    """ScoutingNano schema builder

    ScoutingNano is a NanoAOD format that includes Scouting objects
    """

    mixins = {
        **NanoAODSchema.mixins,
        "ScoutingPFJet": "Jet",
        "ScoutingPFJetRecluster": "Jet",
        "ScoutingFatJet": "Jet",
        "ScoutingMuonNoVtxDisplacedVertex": "Vertex",
        "ScoutingMuonVtxDisplacedVertex": "Vertex",
        "ScoutingElectron": "Electron",
        "ScoutingPhoton": "Photon", 
        "ScoutingMuonNoVtx": "Muon",
        "ScoutingMuonVtx": "Muon"

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
        hists_to_process=[]
    ):
        self.trigger_paths = trigger_paths
        self.hists_to_process = hists_to_process
        
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
        
    def process(self, events):
        dataset = events.metadata['dataset']
        hist_dict = {}
        return_dict = {}
               
        # Saturated-Jets event cut
        events = events[dak.all(events.L1Jet.pt<1000,axis=1)]
        # Saturated-MET event cut
        events = events[dak.flatten(events.L1EtSum.pt[(events.L1EtSum.etSumType==2) 
                                                      & (events.L1EtSum.bx==0)])<1040]
        
        if ("lead_scoutingjet_pt" in self.hists_to_process):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.pt_axis, 
                "lead_scoutingjet_pt"
            )
        if ("lead_scoutingjet_eta" in self.hists_to_process):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.eta_axis, 
                self.pt_axis,
                "lead_scoutingjet_eta" 
            )
        if ("lead_scoutingjet_phi" in self.hists_to_process):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.phi_axis, 
                self.pt_axis,
                "lead_scoutingjet_phi" 
            )
        if ("lead_scoutingjet_eta_phi" in self.hists_to_process):
            hist_dict = createHist_3d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.eta_axis, 
                self.phi_axis,
                self.pt_axis,
                "lead_scoutingjet_eta_phi" 
            )
        if ("lead_scoutingjet_ieta_iphi" in self.hists_to_process):
            hist_dict = createHist_3d(
                hist_dict, 
                self.dataset_axis,
                self.trigger_axis, 
                self.ieta_axis, 
                self.iphi_axis, 
                self.pt_axis,
                "lead_scoutingjet_ieta_iphi" 
            )
        if ("lead_scoutingelectron_pt" in self.hists_to_process):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.pt_axis, 
                "lead_scoutingelectron_pt"
            )
        if ("lead_scoutingelectron_eta" in self.hists_to_process):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.eta_axis, 
                self.pt_axis,
                "lead_scoutingelectron_eta" 
            )
        if ("lead_scoutingelectron_phi" in self.hists_to_process):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.phi_axis, 
                self.pt_axis,
                "lead_scoutingelectron_phi" 
            )
        if ("lead_scoutingelectron_eta_phi" in self.hists_to_process):
            hist_dict = createHist_3d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.eta_axis, 
                self.phi_axis,
                self.pt_axis,
                "lead_scoutingelectron_eta_phi" 
            )
        if ("lead_scoutingelectron_ieta_iphi" in self.hists_to_process):
            hist_dict = createHist_3d(
                hist_dict, 
                self.dataset_axis,
                self.trigger_axis, 
                self.ieta_axis, 
                self.iphi_axis, 
                self.pt_axis,
                "lead_scoutingelectron_ieta_iphi" 
            )
        if ("lead_scoutingphoton_pt" in self.hists_to_process):
            hist_dict = createHist_1d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.pt_axis, 
                "lead_scoutingphoton_pt"
            )
        if ("lead_scoutingphoton_eta" in self.hists_to_process):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.eta_axis, 
                self.pt_axis,
                "lead_scoutingphoton_eta" 
            )
        if ("lead_scoutingphoton_phi" in self.hists_to_process):
            hist_dict = createHist_2d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.phi_axis, 
                self.pt_axis,
                "lead_scoutingphoton_phi" 
            )
        if ("lead_scoutingphoton_eta_phi" in self.hists_to_process):
            hist_dict = createHist_3d(
                hist_dict, 
                self.dataset_axis, 
                self.trigger_axis, 
                self.eta_axis, 
                self.phi_axis,
                self.pt_axis,
                "lead_scoutingphoton_eta_phi" 
            )
        if ("lead_scoutingphoton_ieta_iphi" in self.hists_to_process):
            hist_dict = createHist_3d(
                hist_dict, 
                self.dataset_axis,
                self.trigger_axis, 
                self.ieta_axis, 
                self.iphi_axis, 
                self.pt_axis,
                "lead_scoutingphoton_ieta_iphi" 
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
            
            scoutingjet_mask = (
                #(abs(events_trig.ScoutingPFJetRecluster.eta) < 2.4) &                    # remove forward jets
                (events_trig.ScoutingPFJetRecluster.neHEF < 0.99) &                      # neutral hadron fraction
                (events_trig.ScoutingPFJetRecluster.neEmEF < 0.90) &                     # neutral em fraction
                (events_trig.ScoutingPFJetRecluster.nConstituents > 1) &                 # number of constituents
                (events_trig.ScoutingPFJetRecluster.muEF < 0.80) &                       # muon fraction
                (events_trig.ScoutingPFJetRecluster.chHEF > 0.01) &                      # charged hadron fraction
                (events_trig.ScoutingPFJetRecluster.nCh > 0) &                           # charged multiplicity
                (events_trig.ScoutingPFJetRecluster.chEmEF < 0.80)                       # charged em fraction
            )
            
            selected_scoutingjets_trig = events_trig.ScoutingPFJetRecluster[scoutingjet_mask]
                
            # require at least one valid jet per event
            selected_scoutingjets_trig = selected_scoutingjets_trig[ak.num(selected_scoutingjets_trig,axis=1)>=1]
            selected_scoutingjets_trig.phi_mod = dak.where(
                selected_scoutingjets_trig.phi < -2.967, 
                selected_scoutingjets_trig.phi  + 2 * np.pi, 
                selected_scoutingjets_trig.phi
            )
                                        
            if ("lead_scoutingjet_pt" in self.hists_to_process):
                lead_scoutingjet_pt = selected_scoutingjets_trig.pt[:, 0]
                hist_dict = fillHist_1d(
                    hist_dict, 
                    "lead_scoutingjet_pt", 
                    dataset, 
                    lead_scoutingjet_pt, 
                    "pt", 
                    trigger_path
                )
            if ("lead_scoutingjet_eta" in self.hists_to_process):
                lead_scoutingjet_eta = selected_scoutingjets_trig.eta[:, 0]
                lead_scoutingjet_pt = selected_scoutingjets_trig.pt[:, 0]
                hist_dict = fillHist_2d(
                    hist_dict, 
                    'lead_scoutingjet_eta', 
                    dataset, 
                    lead_scoutingjet_eta, 
                    lead_scoutingjet_pt,
                    "eta",
                    "pt",
                    trigger_path, 
                )
            if ("lead_scoutingjet_phi" in self.hists_to_process):
                lead_scoutingjet_phi = selected_scoutingjets_trig.phi[:, 0]
                lead_scoutingjet_pt = selected_scoutingjets_trig.pt[:, 0]
                hist_dict = fillHist_2d(
                    hist_dict, 
                    'lead_scoutingjet_phi', 
                    dataset, 
                    lead_scoutingjet_phi, 
                    lead_scoutingjet_pt,
                    "phi",
                    "pt",
                    trigger_path, 
                )
            if ("lead_scoutingjet_eta_phi" in self.hists_to_process):
                lead_scoutingjet_eta = selected_scoutingjets_trig.eta[:, 0]
                lead_scoutingjet_phi = selected_scoutingjets_trig.phi[:, 0]
                lead_scoutingjet_pt = selected_scoutingjets_trig.pt[:, 0]
                hist_dict = fillHist_3d(
                    hist_dict, 
                    'lead_scoutingjet_eta_phi', 
                    dataset, 
                    lead_scoutingjet_eta, 
                    lead_scoutingjet_phi,
                    lead_scoutingjet_pt,
                    "eta",
                    "phi",
                    "pt",
                    trigger_path, 
                )
            if ("lead_scoutingjet_ieta_iphi" in self.hists_to_process):
                lead_scoutingjet_eta = selected_scoutingjets_trig.eta[:, 0]
                lead_scoutingjet_phi = selected_scoutingjets_trig.phi_mod[:, 0]
                lead_scoutingjet_pt = selected_scoutingjets_trig.pt[:, 0]
                hist_dict = fillHist_3d(
                    hist_dict, 
                    'lead_scoutingjet_ieta_iphi', 
                    dataset, 
                    lead_scoutingjet_eta, 
                    lead_scoutingjet_phi,
                    lead_scoutingjet_pt,
                    "ieta",
                    "iphi",
                    "pt",
                    trigger_path, 
                )
            
            selected_scoutingelectrons_trig = events_trig.ScoutingElectron[ak.num(events_trig.ScoutingElectron,axis=1)>=1]
            selected_scoutingelectrons_trig.phi_mod = dak.where(
                selected_scoutingelectrons_trig.phi < -2.967, 
                selected_scoutingelectrons_trig.phi  + 2 * np.pi, 
                selected_scoutingelectrons_trig.phi
            )
                                        
            if ("lead_scoutingelectron_pt" in self.hists_to_process):
                lead_scoutingelectron_pt = selected_scoutingelectrons_trig.pt[:, 0]
                hist_dict = fillHist_1d(
                    hist_dict, 
                    "lead_scoutingelectron_pt", 
                    dataset, 
                    lead_scoutingelectron_pt, 
                    "pt", 
                    trigger_path
                )
            if ("lead_scoutingelectron_eta" in self.hists_to_process):
                lead_scoutingelectron_eta = selected_scoutingelectrons_trig.eta[:, 0]
                lead_scoutingelectron_pt = selected_scoutingelectrons_trig.pt[:, 0]
                hist_dict = fillHist_2d(
                    hist_dict, 
                    'lead_scoutingelectron_eta', 
                    dataset, 
                    lead_scoutingelectron_eta, 
                    lead_scoutingelectron_pt,
                    "eta",
                    "pt",
                    trigger_path, 
                )
            if ("lead_scoutingelectron_phi" in self.hists_to_process):
                lead_scoutingelectron_phi = selected_scoutingelectrons_trig.phi[:, 0]
                lead_scoutingelectron_pt = selected_scoutingelectrons_trig.pt[:, 0]
                hist_dict = fillHist_2d(
                    hist_dict, 
                    'lead_scoutingelectron_phi', 
                    dataset, 
                    lead_scoutingelectron_phi, 
                    lead_scoutingelectron_pt,
                    "phi",
                    "pt",
                    trigger_path, 
                )
            if ("lead_scoutingelectron_eta_phi" in self.hists_to_process):
                lead_scoutingelectron_eta = selected_scoutingelectrons_trig.eta[:, 0]
                lead_scoutingelectron_phi = selected_scoutingelectrons_trig.phi[:, 0]
                lead_scoutingelectron_pt = selected_scoutingelectrons_trig.pt[:, 0]
                hist_dict = fillHist_3d(
                    hist_dict, 
                    'lead_scoutingelectron_eta_phi', 
                    dataset, 
                    lead_scoutingelectron_eta, 
                    lead_scoutingelectron_phi,
                    lead_scoutingelectron_pt,
                    "eta",
                    "phi",
                    "pt",
                    trigger_path, 
                )
            if ("lead_scoutingelectron_ieta_iphi" in self.hists_to_process):
                lead_scoutingelectron_eta = selected_scoutingelectrons_trig.eta[:, 0]
                lead_scoutingelectron_phi = selected_scoutingelectrons_trig.phi_mod[:, 0]
                lead_scoutingelectron_pt = selected_scoutingelectrons_trig.pt[:, 0]
                hist_dict = fillHist_3d(
                    hist_dict, 
                    'lead_scoutingelectron_ieta_iphi', 
                    dataset, 
                    lead_scoutingelectron_eta, 
                    lead_scoutingelectron_phi,
                    lead_scoutingelectron_pt,
                    "ieta",
                    "iphi",
                    "pt",
                    trigger_path, 
                )
                
            selected_scoutingphotons_trig = events_trig.ScoutingPhoton[ak.num(events_trig.ScoutingPhoton,axis=1)>=1]
            selected_scoutingphotons_trig.phi_mod = dak.where(
                selected_scoutingphotons_trig.phi < -2.967, 
                selected_scoutingphotons_trig.phi  + 2 * np.pi, 
                selected_scoutingphotons_trig.phi
            )
                                        
            if ("lead_scoutingphoton_pt" in self.hists_to_process):
                lead_scoutingphoton_pt = selected_scoutingphotons_trig.pt[:, 0]
                hist_dict = fillHist_1d(
                    hist_dict, 
                    "lead_scoutingphoton_pt", 
                    dataset, 
                    lead_scoutingphoton_pt, 
                    "pt", 
                    trigger_path
                )
            if ("lead_scoutingphoton_eta" in self.hists_to_process):
                lead_scoutingphoton_eta = selected_scoutingphotons_trig.eta[:, 0]
                lead_scoutingphoton_pt = selected_scoutingphotons_trig.pt[:, 0]
                hist_dict = fillHist_2d(
                    hist_dict, 
                    'lead_scoutingphoton_eta', 
                    dataset, 
                    lead_scoutingphoton_eta, 
                    lead_scoutingphoton_pt,
                    "eta",
                    "pt",
                    trigger_path, 
                )
            if ("lead_scoutingphoton_phi" in self.hists_to_process):
                lead_scoutingphoton_phi = selected_scoutingphotons_trig.phi[:, 0]
                lead_scoutingphoton_pt = selected_scoutingphotons_trig.pt[:, 0]
                hist_dict = fillHist_2d(
                    hist_dict, 
                    'lead_scoutingphoton_phi', 
                    dataset, 
                    lead_scoutingphoton_phi, 
                    lead_scoutingphoton_pt,
                    "phi",
                    "pt",
                    trigger_path, 
                )
            if ("lead_scoutingphoton_eta_phi" in self.hists_to_process):
                lead_scoutingphoton_eta = selected_scoutingphotons_trig.eta[:, 0]
                lead_scoutingphoton_phi = selected_scoutingphotons_trig.phi[:, 0]
                lead_scoutingphoton_pt = selected_scoutingphotons_trig.pt[:, 0]
                hist_dict = fillHist_3d(
                    hist_dict, 
                    'lead_scoutingphoton_eta_phi', 
                    dataset, 
                    lead_scoutingphoton_eta, 
                    lead_scoutingphoton_phi,
                    lead_scoutingphoton_pt,
                    "eta",
                    "phi",
                    "pt",
                    trigger_path, 
                )
            if ("lead_scoutingphoton_ieta_iphi" in self.hists_to_process):
                lead_scoutingphoton_eta = selected_scoutingphotons_trig.eta[:, 0]
                lead_scoutingphoton_phi = selected_scoutingphotons_trig.phi_mod[:, 0]
                lead_scoutingphoton_pt = selected_scoutingphotons_trig.pt[:, 0]
                hist_dict = fillHist_3d(
                    hist_dict, 
                    'lead_scoutingphoton_ieta_iphi', 
                    dataset, 
                    lead_scoutingphoton_eta, 
                    lead_scoutingphoton_phi,
                    lead_scoutingphoton_pt,
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
    for j in range(270,278):
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
