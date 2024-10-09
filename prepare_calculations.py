import pathlib

import custom_tasks as ct
import numpy as np

import fireworks as fw

###############################################################################
# Supporting electrolyte cations calculation script
###############################################################################
# This script prepares the calculations for the cations. It is based on the
# fireworks package, which is responsible for managing the calculations.
# The various programs are loaded via environment modules, which can be
# specified together with other settings in the following sections.
# The scripts are meant to be executed on a system running with the
# SLURM workload manager (https://slurm.schedmd.com)
###############################################################################

# Settings of MongoDB database for fireworks
launchpad = fw.LaunchPad(
    host="ip-adress",
    port="port",
    name="database-name",
    username="database-user",
    password="database-password",
)

# File containing the PubChem CIDs of the cations to calculate
cid_file = "cids.txt"

# Path to directory including the data used for starting the calculations
# For each cation there has to be a folder named after the PubChem CID
# containing a "CID.xyz" file including the structure and ".CHRG" file
# including the charge.
data_dir = "/path/to/data/directory"

# Path to the directory, where the computations are evaluated.
# Usually on a local disk of the computation nodes
calc_dir = "/path/to/calculation/directory"

# Templates for Orca and Turbomole
orca_elprop_t = "template_orca_b973c_elprop.txt"
orca_geo_t = "template_orca_b973c_geoopt.txt"
orca_freq_t = "template_orca_b973c_freq.txt"
orca_sp_t = "template_orca_b973c_sp.txt"
turbomole_t = "template_tmol_bp_def2_TZVPD.txt"
turbomole_cosmo_t = "cosmo.txt"
cosmotherm_gsolv_t = "gsolv.input"
cosmotherm_logp_t = "logp.input"

# Names of environment modules of the programs
orca_module = "orca/5.0.3"
xtb_module = "xtb/6.5.1"
crest_module = "crest/2.12"
turbomole_module = "turbomole/6.5/intel"
turbomole_mpi_module = "turbomole/6.5/parallel_mpi"
cosmotherm_module = "cosmotherm/2021"

# Number of CPUs for the various programs
orca_ncpu = 4
xtb_ncpu = 4
crest_ncpu = 32
tmol_ncpu = 4

# Memory settings
orca_membase = 12000  # Value is in MB and per CPU
crest_mem = "50G"
turbomole_mem = "30G"

# Disk space on nodes
tmp = "100G"

# Metadata for workflow. Makes filtering of fireworks workflows easier when
# using it also for other calculations
metadata = {"molecule_type": "cation"}

###############################################################################

# Start of the preparation steps
# Templates for the input files of the Orca calculations
pwd = pathlib.Path.cwd()

orca_elprop_task = ct.OrcaTemplateWriter(
    {
        "template_file": str(pwd / "templates" / orca_elprop_t),
        "output_file": "orca.inp",
    }
)

orca_geo_task = ct.OrcaTemplateWriter(
    {
        "template_file": str(pwd / "templates" / orca_geo_t),
        "output_file": "orca.inp",
    }
)

orca_freq_task = ct.OrcaTemplateWriter(
    {
        "template_file": str(pwd / "templates" / orca_freq_t),
        "output_file": "orca.inp",
    }
)

orca_sp_task = ct.OrcaTemplateWriter(
    {
        "template_file": str(pwd / "templates" / orca_sp_t),
        "output_file": "orca.inp",
    }
)

# Setting up a workflow for each cation
with open(cid_file, "r") as cid_file:
    for line in cid_file:
        cid = line.strip()
        input_fw = fw.Firework(
            ct.InputCollection(),
            spec={
                "type": "input",
                "cid": cid,
                "data_dir": data_dir,
                "xtb_ncpu": xtb_ncpu,
                "crest_ncpu": crest_ncpu,
                "_files_out": {
                    "conf_xyz": str(
                        pathlib.Path(data_dir) / cid / "Crest" / "crest_best.xyz"
                    )
                },
                "_launch_dir": str(pathlib.Path(data_dir) / cid),
            },
            name="start_" + cid,
        )

        geo_fw = fw.Firework(
            [
                ct.XTBGeometryOptimization(),
                ct.XtbOut2OrcaIn(),
                orca_geo_task,
                ct.OrcaCalculations(),
                ct.OrcaEnergyParser(),
            ],
            spec={
                "type": "geo",
                "xtb_ncpu": "1",
                "orca_ncpu": orca_ncpu,
                "orca_membase": orca_membase,
                "_files_in": {"conf_xyz": "mol.xyz"},
                "_files_out": {"xyz": "orca.xyz"},
                "_launch_dir": str(pathlib.Path(data_dir) / cid / "geo"),
                "_queueadapter": {
                    "cpus_per_task": "1",
                    "ntasks": orca_ncpu,
                    "ntasks_per_node": orca_ncpu,
                    "mem": str(int(np.round(orca_membase * orca_ncpu / 1000, -1)))
                    + "G",
                    "tmp": tmp,
                    "orca": orca_module,
                    "xtb": xtb_module,
                },
            },
            name="geo_" + cid,
        )
        geo_ox_fw = fw.Firework(
            [
                ct.XTBGeometryOptimization(),
                ct.XtbOut2OrcaIn(),
                orca_geo_task,
                ct.OrcaCalculations(),
            ],
            spec={
                "type": "geo_ox",
                "xtb_ncpu": "1",
                "orca_ncpu": orca_ncpu,
                "orca_membase": orca_membase,
                "ox": True,
                "_files_in": {"xyz": "mol.xyz"},
                "_files_out": {"ox_xyz": "orca.xyz"},
                "_launch_dir": str(pathlib.Path(data_dir) / cid / "geo_oxidation"),
                "_queueadapter": {
                    "cpus_per_task": "1",
                    "ntasks": orca_ncpu,
                    "ntasks_per_node": orca_ncpu,
                    "mem": str(int(np.round(orca_membase * orca_ncpu / 1000, -1)))
                    + "G",
                    "tmp": tmp,
                    "orca": orca_module,
                    "xtb": xtb_module,
                },
            },
            name="geo_ox_" + cid,
        )
        geo_red_fw = fw.Firework(
            [
                ct.XTBGeometryOptimization(),
                ct.XtbOut2OrcaIn(),
                orca_geo_task,
                ct.OrcaCalculations(),
                ct.OrcaEnergyParser(),
            ],
            spec={
                "type": "geo_red",
                "xtb_ncpu": "1",
                "orca_ncpu": orca_ncpu,
                "orca_membase": orca_membase,
                "red": True,
                "_files_in": {"conf_red_xyz": "mol.xyz"},
                "_files_out": {"red_xyz": "orca.xyz"},
                "_launch_dir": str(pathlib.Path(data_dir) / cid / "geo_reduction"),
                "_queueadapter": {
                    "cpus_per_task": "1",
                    "ntasks": orca_ncpu,
                    "ntasks_per_node": orca_ncpu,
                    "mem": str(int(np.round(orca_membase * orca_ncpu / 1000, -1)))
                    + "G",
                    "tmp": tmp,
                    "orca": orca_module,
                    "xtb": xtb_module,
                },
            },
            name="geo_red_" + cid,
        )

        crestconf_red_fw = fw.Firework(
            [ct.CrestConformationSearch(), ct.CrestFragmentParser()],
            spec={
                "crest_ncpu": crest_ncpu,
                "red": True,
                "type": "crestconf_red",
                "_files_in": {"xyz": "mol.xyz"},
                "_files_out": {"conf_red_xyz": "crest_best.xyz"},
                "_launch_dir": str(pathlib.Path(data_dir) / cid / "Crest_reduction"),
                "_queueadapter": {
                    "cpus_per_task": crest_ncpu,
                    "ntasks": "1",
                    "ntasks_per_node": "1",
                    "crest": crest_module,
                    "mem": crest_mem,
                    "tmp": tmp,
                },
            },
            name="crest_red_" + cid,
        )

        freq_fw = fw.Firework(
            [orca_freq_task, ct.OrcaCalculations(), ct.OrcaFreqeuncyParser()],
            spec={
                "type": "freq",
                "orca_ncpu": orca_ncpu,
                "orca_membase": orca_membase,
                "_files_in": {"xyz": "mol.xyz"},
                "_launch_dir": str(pathlib.Path(data_dir) / cid / "freq"),
                "_queueadapter": {
                    "cpus_per_task": "1",
                    "ntasks": orca_ncpu,
                    "ntasks_per_node": orca_ncpu,
                    "mem": str(int(np.round(orca_membase * orca_ncpu / 1000, -1)))
                    + "G",
                    "tmp": tmp,
                    "orca": orca_module,
                },
            },
            name="freq_" + cid,
        )
        freq_ox_fw = fw.Firework(
            [orca_freq_task, ct.OrcaCalculations(), ct.OrcaFreqeuncyParser()],
            spec={
                "type": "freq_ox",
                "orca_ncpu": orca_ncpu,
                "orca_membase": orca_membase,
                "ox": True,
                "_files_in": {"ox_xyz": "mol.xyz"},
                "_launch_dir": str(pathlib.Path(data_dir) / cid / "freq_oxidation"),
                "_queueadapter": {
                    "cpus_per_task": "1",
                    "ntasks": orca_ncpu,
                    "ntasks_per_node": orca_ncpu,
                    "mem": str(int(np.round(orca_membase * orca_ncpu / 1000, -1)))
                    + "G",
                    "tmp": tmp,
                    "orca": orca_module,
                },
            },
            name="freq_ox_" + cid,
        )
        freq_red_fw = fw.Firework(
            [orca_freq_task, ct.OrcaCalculations(), ct.OrcaFreqeuncyParser()],
            spec={
                "type": "freq_red",
                "orca_ncpu": orca_ncpu,
                "orca_membase": orca_membase,
                "red": True,
                "_files_in": {"red_xyz": "mol.xyz"},
                "_launch_dir": str(pathlib.Path(data_dir) / cid / "freq_reduction"),
                "_queueadapter": {
                    "cpus_per_task": "1",
                    "ntasks": orca_ncpu,
                    "ntasks_per_node": orca_ncpu,
                    "mem": str(int(np.round(orca_membase * orca_ncpu / 1000, -1)))
                    + "G",
                    "tmp": tmp,
                    "orca": orca_module,
                },
            },
            name="freq_red_" + cid,
        )

        elprop_fw = fw.Firework(
            [
                orca_elprop_task,
                ct.OrcaCalculations(),
                ct.OrcaOrbitalParser(),
                ct.OrcaElectronicParser(),
            ],
            spec={
                "type": "elprop",
                "orca_ncpu": orca_ncpu,
                "orca_membase": orca_membase,
                "_files_in": {"xyz": "mol.xyz"},
                "_launch_dir": str(pathlib.Path(data_dir) / cid / "elprop"),
                "_queueadapter": {
                    "cpus_per_task": "1",
                    "ntasks": orca_ncpu,
                    "ntasks_per_node": orca_ncpu,
                    "mem": str(int(np.round(orca_membase * orca_ncpu / 1000, -1)))
                    + "G",
                    "tmp": tmp,
                    "orca": orca_module,
                },
            },
            name="elprop_" + cid,
        )

        elprop_red_fw = fw.Firework(
            [
                orca_elprop_task,
                ct.OrcaCalculations(),
                ct.OrcaOrbitalParser(),
                ct.OrcaElectronicParser(),
            ],
            spec={
                "type": "elprop_red",
                "orca_ncpu": orca_ncpu,
                "orca_membase": orca_membase,
                "red": True,
                "_files_in": {"red_xyz": "mol.xyz"},
                "_launch_dir": str(pathlib.Path(data_dir) / cid / "elprop_reduction"),
                "_queueadapter": {
                    "cpus_per_task": "1",
                    "ntasks": orca_ncpu,
                    "ntasks_per_node": orca_ncpu,
                    "mem": str(int(np.round(orca_membase * orca_ncpu / 1000, -1)))
                    + "G",
                    "tmp": tmp,
                    "orca": orca_module,
                },
            },
            name="elprop_red_" + cid,
        )

        E_red_neutral_fw = fw.Firework(
            [orca_sp_task, ct.OrcaCalculations(), ct.OrcaEnergyParser()],
            spec={
                "type": "E_red_neutral",
                "orca_ncpu": orca_ncpu,
                "orca_membase": orca_membase,
                "_files_in": {"red_xyz": "mol.xyz"},
                "_launch_dir": str(
                    pathlib.Path(data_dir) / cid / "sp_reduction_neutral"
                ),
                "_queueadapter": {
                    "cpus_per_task": "1",
                    "ntasks": orca_ncpu,
                    "ntasks_per_node": orca_ncpu,
                    "mem": str(int(np.round(orca_membase * orca_ncpu / 1000, -1)))
                    + "G",
                    "tmp": tmp,
                    "orca": orca_module,
                },
            },
            name="sp_red_n_" + cid,
        )

        E_neutral_red_fw = fw.Firework(
            [orca_sp_task, ct.OrcaCalculations(), ct.OrcaEnergyParser()],
            spec={
                "type": "E_neutral_red",
                "orca_ncpu": orca_ncpu,
                "orca_membase": orca_membase,
                "red": True,
                "_files_in": {"xyz": "mol.xyz"},
                "_launch_dir": str(pathlib.Path(data_dir) / cid / "sp_neutral_reduced"),
                "_queueadapter": {
                    "cpus_per_task": "1",
                    "ntasks": orca_ncpu,
                    "ntasks_per_node": orca_ncpu,
                    "mem": str(int(np.round(orca_membase * orca_ncpu / 1000, -1)))
                    + "G",
                    "tmp": tmp,
                    "orca": orca_module,
                },
            },
            name="sp_neutral_r_" + cid,
        )

        turbo_vacuum_fw = fw.Firework(
            [ct.TurbomoleDefine(), ct.TurbomoleCalculations()],
            spec={
                "type": "turbo_vacuum",
                "tmol_template": str(pwd / "templates" / turbomole_t),
                "_files_in": {"xyz": "mol.xyz"},
                "_files_out": {"tmol_energy": "out.energy"},
                "_launch_dir": str(pathlib.Path(data_dir) / cid / "cosmo" / "vacuum"),
                "_queueadapter": {
                    "cpus_per_task": tmol_ncpu,
                    "ntasks": "1",
                    "ntasks_per_node": "1",
                    "mem": turbomole_mem,
                    "tmp": tmp,
                    "turbomole": turbomole_mpi_module,
                },
            },
            name="tmol_vacuum_" + cid,
        )
        turbo_infinity_fw = fw.Firework(
            [ct.TurbomoleDefine(), ct.TurbomoleCosmoprep(), ct.TurbomoleCalculations()],
            spec={
                "type": "turbo_infinity",
                "tmol_template": str(pwd / "templates" / turbomole_t),
                "tmol_cosmo_template": str(pwd / "templates" / turbomole_cosmo_t),
                "_files_in": {"xyz": "mol.xyz"},
                "_files_out": {"tmol_cosmo": "out.cosmo"},
                "_launch_dir": str(pathlib.Path(data_dir) / cid / "cosmo" / "infinity"),
                "_queueadapter": {
                    "cpus_per_task": "1",
                    "ntasks": "1",
                    "ntasks_per_node": "1",
                    "mem": turbomole_mem,
                    "tmp": tmp,
                    "turbomole": turbomole_module,
                },
            },
            name="tmol_infinity_" + cid,
        )
        cosmotherm_fw = fw.Firework(
            [
                ct.CosmothermCalculations(),
                ct.CosmothermCommonParser(),
                ct.CosmothermGsolvParser(),
                ct.CosmothermLogpParser(),
            ],
            spec={
                "type": "cosmotherm",
                "cosmotherm_gsolv_template": str(pwd / "templates" / cosmotherm_gsolv_t),
                "cosmotherm_logp_template": str(pwd / "templates" / cosmotherm_logp_t),
                "_files_in": {"tmol_energy": "mol.energy", "tmol_cosmo": "mol.cosmo"},
                "_launch_dir": str(
                    pathlib.Path(data_dir) / cid / "cosmo" / "cosmotherm"
                ),
                "_queueadapter": {
                    "turbomole": turbomole_module,
                    "cosmotherm": cosmotherm_module,
                },
            },
            name="cosmotherm_" + cid,
        )

        turbo_vacuum_ox_fw = fw.Firework(
            [ct.TurbomoleDefine(), ct.TurbomoleCalculations()],
            spec={
                "type": "turbo_vacuum_ox",
                "ox": True,
                "tmol_template": str(pwd / "templates" / turbomole_t),
                "_files_in": {"ox_xyz": "mol.xyz"},
                "_files_out": {"ox_tmol_energy": "out.energy"},
                "_launch_dir": str(
                    pathlib.Path(data_dir) / cid / "cosmo_oxidation" / "vacuum"
                ),
                "_queueadapter": {
                    "cpus_per_task": tmol_ncpu,
                    "ntasks": "1",
                    "ntasks_per_node": "1",
                    "mem": turbomole_mem,
                    "tmp": tmp,
                    "turbomole": turbomole_mpi_module,
                },
            },
            name="tmol_vacuum_ox_" + cid,
        )
        turbo_infinity_ox_fw = fw.Firework(
            [ct.TurbomoleDefine(), ct.TurbomoleCosmoprep(), ct.TurbomoleCalculations()],
            spec={
                "type": "turbo_infinity_ox",
                "ox": True,
                "tmol_template": str(pwd / "templates" / turbomole_t),
                "tmol_cosmo_template": str(pwd / "templates" / turbomole_cosmo_t),
                "_files_in": {"ox_xyz": "mol.xyz"},
                "_files_out": {"ox_tmol_cosmo": "out.cosmo"},
                "_launch_dir": str(
                    pathlib.Path(data_dir) / cid / "cosmo_oxidation" / "infinity"
                ),
                "_queueadapter": {
                    "cpus_per_task": "1",
                    "ntasks": "1",
                    "ntasks_per_node": "1",
                    "mem": turbomole_mem,
                    "tmp": tmp,
                    "turbomole": turbomole_module,
                },
            },
            name="tmol_infinity_ox_" + cid,
        )
        cosmotherm_ox_fw = fw.Firework(
            [ct.CosmothermCalculations(), ct.CosmothermGsolvParser()],
            spec={
                "type": "cosmotherm_ox",
                "ox": True,
                "cosmotherm_gsolv_template": str(pwd / "templates" / cosmotherm_gsolv_t),
                "_files_in": {
                    "ox_tmol_energy": "mol.energy",
                    "ox_tmol_cosmo": "mol.cosmo",
                },
                "_launch_dir": str(
                    pathlib.Path(data_dir) / cid / "cosmo_oxidation" / "cosmotherm"
                ),
                "_queueadapter": {
                    "turbomole": turbomole_module,
                    "cosmotherm": cosmotherm_module,
                },
            },
            name="cosmotherm_ox_" + cid,
        )

        turbo_vacuum_red_fw = fw.Firework(
            [ct.TurbomoleDefine(), ct.TurbomoleCalculations()],
            spec={
                "type": "turbo_vacuum_red",
                "red": True,
                "tmol_template": str(pwd / "templates" / turbomole_t),
                "_files_in": {"red_xyz": "mol.xyz"},
                "_files_out": {"red_tmol_energy": "out.energy"},
                "_launch_dir": str(
                    pathlib.Path(data_dir) / cid / "cosmo_reduction" / "vacuum"
                ),
                "_queueadapter": {
                    "cpus_per_task": tmol_ncpu,
                    "ntasks": "1",
                    "ntasks_per_node": "1",
                    "mem": turbomole_mem,
                    "tmp": tmp,
                    "turbomole": turbomole_mpi_module,
                },
            },
            name="tmol_vacuum_red_" + cid,
        )
        turbo_infinity_red_fw = fw.Firework(
            [ct.TurbomoleDefine(), ct.TurbomoleCosmoprep(), ct.TurbomoleCalculations()],
            spec={
                "type": "turbo_infinity_red",
                "red": True,
                "tmol_template": str(pwd / "templates" / turbomole_t),
                "tmol_cosmo_template": str(pwd / "templates" / turbomole_cosmo_t),
                "_files_in": {"red_xyz": "mol.xyz"},
                "_files_out": {"red_tmol_cosmo": "out.cosmo"},
                "_launch_dir": str(
                    pathlib.Path(data_dir) / cid / "cosmo_reduction" / "infinity"
                ),
                "_queueadapter": {
                    "cpus_per_task": "1",
                    "ntasks": "1",
                    "ntasks_per_node": "1",
                    "mem": turbomole_mem,
                    "tmp": tmp,
                    "turbomole": turbomole_module,
                },
            },
            name="tmol_infinity_red_" + cid,
        )
        cosmotherm_red_fw = fw.Firework(
            [ct.CosmothermCalculations(), ct.CosmothermGsolvParser()],
            spec={
                "type": "cosmotherm_red",
                "red": True,
                "cosmotherm_gsolv_template": str(pwd / "templates" / cosmotherm_gsolv_t),
                "_files_in": {
                    "red_tmol_energy": "mol.energy",
                    "red_tmol_cosmo": "mol.cosmo",
                },
                "_launch_dir": str(
                    pathlib.Path(data_dir) / cid / "cosmo_reduction" / "cosmotherm"
                ),
                "_queueadapter": {
                    "turbomole": turbomole_module,
                    "cosmotherm": cosmotherm_module,
                },
            },
            name="cosmotherm_red_" + cid,
        )

        wf = fw.Workflow(
            [
                input_fw,
                geo_fw,
                elprop_fw,
                freq_fw,
                E_neutral_red_fw,
                turbo_vacuum_fw,
                turbo_infinity_fw,
                cosmotherm_fw,
                geo_ox_fw,
                freq_ox_fw,
                turbo_vacuum_ox_fw,
                turbo_infinity_ox_fw,
                cosmotherm_ox_fw,
                geo_red_fw,
                elprop_red_fw,
                freq_red_fw,
                crestconf_red_fw,
                E_red_neutral_fw,
                turbo_vacuum_red_fw,
                turbo_infinity_red_fw,
                cosmotherm_red_fw,
            ],
            {
                input_fw: [geo_fw],
                geo_fw: [
                    elprop_fw,
                    freq_fw,
                    E_neutral_red_fw,
                    geo_ox_fw,
                    crestconf_red_fw,
                    turbo_vacuum_fw,
                    turbo_infinity_fw,
                ],
                geo_ox_fw: [freq_ox_fw, turbo_vacuum_ox_fw, turbo_infinity_ox_fw],
                geo_red_fw: [
                    elprop_red_fw,
                    freq_red_fw,
                    E_red_neutral_fw,
                    turbo_vacuum_red_fw,
                    turbo_infinity_red_fw,
                ],
                crestconf_red_fw: [geo_red_fw],
                turbo_vacuum_fw: [cosmotherm_fw],
                turbo_infinity_fw: [cosmotherm_fw],
                turbo_vacuum_ox_fw: [cosmotherm_ox_fw],
                turbo_infinity_ox_fw: [cosmotherm_ox_fw],
                turbo_vacuum_red_fw: [cosmotherm_red_fw],
                turbo_infinity_red_fw: [cosmotherm_red_fw],
            },
            name=cid,
            metadata=metadata,
        )

        launchpad.add_wf(wf)
