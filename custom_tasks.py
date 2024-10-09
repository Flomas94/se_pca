import fileinput
import glob
import os
import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np
from fireworks.core.firework import FireTaskBase, Firework, FWAction
from jinja2 import Environment, FileSystemLoader
from openbabel import pybel
from prepare_calculations import calc_dir


class NodeCopy:
    def __init__(self, input_dir=None):
        if input_dir:
            self.input_dir = pathlib.Path(input_dir).resolve()
        else:
            self.input_dir = pathlib.Path.cwd()
        self.cwd = pathlib.Path.cwd()
        self.job_id = os.environ.get("SLURM_JOB_ID")
        self.node_dir = pathlib.Path(calc_dir) / self.job_id

    def put(self):
        # os.makedirs(self.node_dir)
        shutil.copytree(
            self.input_dir, self.node_dir, ignore=shutil.ignore_patterns("job.*")
        )
        os.chdir(self.node_dir)

    def get(self):
        ignore = [
            f.parts[-1]
            for f in pathlib.Path.cwd().rglob("*.*")
            if f.is_file() and os.path.getsize(f) > 1073741824
        ]
        if not ignore:
            ignore = None
        else:
            ignore = shutil.ignore_patterns(*ignore)
        shutil.copytree(
            self.node_dir, self.input_dir, dirs_exist_ok=True, ignore=ignore
        )
        shutil.rmtree(self.node_dir)
        os.chdir(self.cwd)


class InputCollection(FireTaskBase):
    _fw_name = "InputCollection"

    def run_task(self, fw_spec):
        cid = fw_spec["cid"]
        data_dir = pathlib.Path(fw_spec["data_dir"])
        cid_dir = data_dir / cid
        if os.path.exists(cid_dir / ".CHRG"):
            with open(cid_dir / ".CHRG", "r") as f:
                input_dict = {"charge": f.readline().strip()}
        else:
            raise FileNotFoundError(f"The .CHRG file is missing for cid {cid}")

        input_dict["multiplicity"] = "1"
        input_dict["cid"] = cid
        input_dict["data_dir"] = fw_spec["data_dir"]

        if os.path.exists(cid_dir / "Crest" / "crest_best.xyz"):
            smiles = (
                next(pybel.readfile("xyz", str(cid_dir / "Crest" / "crest_best.xyz")))
                .write("smi", opt={"n": None})
                .strip()
            )
            smiles = (
                smiles.replace("[N]", "[N+]")
                .replace("[N@]", "[N@+]")
                .replace("[N@@]", "[N@@+]")
            )
            smiles = (
                smiles.replace("[NH]", "[NH+]")
                .replace("[N@H]", "[N@H+]")
                .replace("[N@@H]", "[N@@H+]")
            )
            smiles = (
                smiles.replace("[NH2]", "[NH2+]")
                .replace("[N@H2]", "[N@H2+]")
                .replace("[N@@H2]", "[N@@H2+]")
            )
            smiles = (
                smiles.replace("[NH3]", "[NH3+]")
                .replace("[N@H3]", "[N@H3+]")
                .replace("[N@@H3]", "[N@@H3+]")
            )
            return FWAction(
                stored_data={"charge": input_dict["charge"], "smiles": smiles},
                update_spec=input_dict,
                propagate=True,
            )

        elif os.path.exists(cid_dir / f"{cid}.xyz"):
            smiles = (
                next(pybel.readfile("xyz", str(cid_dir / f"{cid}.xyz")))
                .write("smi", opt={"n": None})
                .strip()
                .replace("[N]", "[N+]")
            )
            smiles = (
                smiles.replace("[N]", "[N+]")
                .replace("[N@]", "[N@+]")
                .replace("[N@@]", "[N@@+]")
            )
            smiles = (
                smiles.replace("[NH]", "[NH+]")
                .replace("[N@H]", "[N@H+]")
                .replace("[N@@H]", "[N@@H+]")
            )
            smiles = (
                smiles.replace("[NH2]", "[NH2+]")
                .replace("[N@H2]", "[N@H2+]")
                .replace("[N@@H2]", "[N@@H2+]")
            )
            smiles = (
                smiles.replace("[NH3]", "[NH3+]")
                .replace("[N@H3]", "[N@H3+]")
                .replace("[N@@H3]", "[N@@H3+]")
            )
            xtb_ncpu = fw_spec["xtb_ncpu"]
            crest_ncpu = fw_spec["crest_ncpu"]
            xtbgeo_wf = Firework(
                [XTBGeometryOptimization(), XTB2Crest()],
                spec={
                    "type": "xtb_geo",
                    "xtb_ncpu": xtb_ncpu,
                    "crest_ncpu": crest_ncpu,
                    "charge": input_dict["charge"],
                    "multiplicity": input_dict["multiplicity"],
                    "data_dir": input_dict["data_dir"],
                    "cid": input_dict["cid"],
                    "_files_prev": {
                        "raw_xyz": str(pathlib.Path(data_dir) / cid / f"{cid}.xyz")
                    },
                    "_files_in": {"raw_xyz": "mol.xyz"},
                    "_files_out": {"xtb_geo_xyz": "xtbopt.xyz"},
                    "_launch_dir": str(pathlib.Path(data_dir) / cid / "xtb"),
                    "_queueadapter": {"cpus_per_task": xtb_ncpu, "xtb": "xtb/6.5.1"},
                },
                name="xtb_geo_" + cid,
            )
            return FWAction(
                stored_data={"charge": input_dict["charge"], "smiles": smiles},
                update_spec=input_dict,
                detours=xtbgeo_wf,
                propagate=True,
            )
        else:
            raise FileNotFoundError(f"No structure file found for cid {cid}.")


class XTBGeometryOptimization(FireTaskBase):
    _fw_name = "XTBGeometryOptimization"

    def run_task(self, fw_spec):
        node_copy = NodeCopy()
        node_copy.put()

        charge = fw_spec["charge"]
        nab = 0
        if "ox" in fw_spec:
            charge = str(int(charge) + 1)
            nab = 1
        elif "red" in fw_spec:
            charge = str(int(charge) - 1)
            nab = 1

        if "gfn" in fw_spec:
            gfn = fw_spec["gfn"]
        else:
            gfn = "2"

        ncpu = fw_spec["xtb_ncpu"]
        stdout_file = pathlib.Path("out.txt").resolve()
        stderr_file = pathlib.Path("err.txt").resolve()

        stdout = subprocess.PIPE
        stderr = subprocess.PIPE

        p = subprocess.Popen(
            f"xtb mol.xyz -c {charge} -u {nab} --opt extreme -P {ncpu} --cycles 4000 --gfn {gfn}",
            stdout=stdout,
            stderr=stderr,
            shell=True,
        )
        (stdout, stderr) = p.communicate()
        returncode = p.returncode

        stdout = stdout.decode("utf-8") if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode("utf-8") if isinstance(stderr, bytes) else stderr

        with open(stdout_file, "a+") as f:
            f.write(stdout)
        with open(stderr_file, "a+") as f:
            f.write(stderr)

        output = {}
        output["returncode"] = returncode

        node_copy.get()

        with open("out.txt", "r") as f:
            failed = False
            for line in f:
                if re.search("FAILED TO CONVERGE GEOMETRY OPTIMIZATION IN", line):
                    failed = True
            if failed:
                raise RuntimeError(
                    "XTBGeometryOptimization fizzled! Not reaching stationary point."
                )

        if returncode != 0:
            raise RuntimeError(
                f"XTBGeometryOptimization fizzled! Return code: {returncode}"
            )

        return FWAction(stored_data=output)


class XTB2Crest(FireTaskBase):
    _fw_name = "XTBOutputHandling"

    def run_task(self, fw_spec):
        data_dir = pathlib.Path(fw_spec["data_dir"])
        cid = fw_spec["cid"]
        crest_ncpu = fw_spec["crest_ncpu"]
        crestconf_wf = Firework(
            CrestConformationSearch(),
            spec={
                "type": "crestconf",
                "crest_ncpu": crest_ncpu,
                "charge": fw_spec["charge"],
                "multiplicity": fw_spec["multiplicity"],
                "data_dir": data_dir,
                "cid": cid,
                "_files_prev": {
                    "xtb_geo_xyz": str(
                        pathlib.Path(data_dir) / cid / "xtb" / "xtbopt.xyz"
                    )
                },
                "_files_in": {"xtb_geo_xyz": "mol.xyz"},
                "_files_out": {"conf_xyz": "crest_best.xyz"},
                "_launch_dir": str(pathlib.Path(data_dir) / cid / "Crest"),
                "_queueadapter": {
                    "cpus_per_task": crest_ncpu,
                    "crest": "crest/2.12",
                    "mem": "50G",
                },
            },
            name="crest_" + cid,
        )
        return FWAction(detours=crestconf_wf, propagate=True)


class CrestConformationSearch(FireTaskBase):
    _fw_name = "CrestConformationSearch"

    def run_task(self, fw_spec):
        node_copy = NodeCopy()
        node_copy.put()

        charge = fw_spec["charge"]
        uhf = "0"
        top = ""
        if "ox" in fw_spec:
            charge = str(int(charge) + 1)
            uhf = "1"
            top = "--noreftopo"
        elif "red" in fw_spec:
            charge = str(int(charge) - 1)
            uhf = "1"
            top = "--noreftopo"

        if "ewin" in fw_spec:
            ewin = f'--ewin {fw_spec["ewin"]}'
        else:
            ewin = ""

        if "clight" in fw_spec:
            clight = "--clight"
        else:
            clight = ""

        ncpu = fw_spec["crest_ncpu"]

        stdout_file = pathlib.Path("out.txt").resolve()
        stderr_file = pathlib.Path("err.txt").resolve()

        stdout = subprocess.PIPE
        stderr = subprocess.PIPE

        p = subprocess.Popen(
            f"crest mol.xyz --chrg {charge} --uhf {uhf} --opt vtight --T {ncpu} --nocross {top} {clight} {ewin}",
            stdout=stdout,
            stderr=stderr,
            shell=True,
        )
        (stdout, stderr) = p.communicate()
        returncode = p.returncode

        stdout = stdout.decode("utf-8") if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode("utf-8") if isinstance(stderr, bytes) else stderr

        with open(stdout_file, "a+") as f:
            f.write(stdout)
        with open(stderr_file, "a+") as f:
            f.write(stderr)

        output = {}
        output["returncode"] = returncode

        if os.path.exists("PROP"):
            shutil.rmtree("PROP", ignore_errors=True)
        # if os.path.exists('crest_conformers.xyz'):
        #    os.remove('crest_conformers.xyz')
        if os.path.exists("crest_rotamers.xyz"):
            os.remove("crest_rotamers.xyz")
        if glob.glob("crest_rotamers*"):
            for junk_file in glob.glob("crest_rotamers*"):
                os.remove(junk_file)
        if os.path.exists("gfnff_topo"):
            os.remove("gfnff_topo")
        if glob.glob("*.tmp"):
            for junk_file in glob.glob("*.tmp"):
                os.remove(junk_file)
        if os.path.exists("crest_property.xyz"):
            os.remove("crest_property.xyz")

        node_copy.get()

        if returncode != 0:
            raise RuntimeError(
                f"CrestConformationSearch fizzled! Return code: {returncode}"
            )

        return FWAction(stored_data=output)


class OrcaOriginParser(FireTaskBase):
    _fw_name = "OrcaOriginParser"

    def run_task(self, fw_spec):
        with open("mol.xyz", "r") as f:
            for line in f:
                if re.match(r"\s N", line):
                    xyz = line.split()
                    xyz = f"{xyz[1]},{xyz[2]},{xyz[3]}"
        return FWAction(update_spec={"orca_origin": xyz})


class OrcaTemplateWriter(FireTaskBase):
    _fw_name = "OrcaTemplateWriter"

    def run_task(self, fw_spec):
        self._load_params(self, fw_spec)

        with open(self.template_file) as f:
            t = Environment(
                loader=FileSystemLoader(self.template_dir), autoescape=True
            ).from_string(f.read())
            output = t.render(self.context)

            write_mode = "w+" if self.append_file else "w"
            with open(self.output_file, write_mode) as of:
                of.write(output)

    def _load_params(self, d_task, d_work):
        self.context = d_work
        if "ox" in d_work:
            self.context["charge"] = str(int(d_work["charge"]) + 1)
            self.context["multiplicity"] = str(2)
        elif "red" in d_work:
            self.context["charge"] = str(int(d_work["charge"]) - 1)
            self.context["multiplicity"] = str(2)
        self.output_file = d_task["output_file"]
        self.append_file = d_task.get("append")  # append to output file?

        self.template_file = d_task["template_file"]
        if not os.path.exists(self.template_file):
            raise ValueError(
                f"TemplateWriterTask could not find a template file at: {self.template_file}"
            )


class OrcaCalculations(FireTaskBase):
    _fw_name = "OrcaCalculations"

    def run_task(self, fw_spec):
        node_copy = NodeCopy()
        node_copy.put()

        stdout_file = pathlib.Path("out.txt").resolve()
        stderr_file = pathlib.Path("err.txt").resolve()

        stdout = subprocess.PIPE
        stderr = subprocess.PIPE

        orcapath = os.environ.get("ORCAPATH")

        p = subprocess.Popen(
            f"{orcapath}/orca orca.inp", stdout=stdout, stderr=stderr, shell=True
        )
        (stdout, stderr) = p.communicate()
        returncode = p.returncode

        stdout = stdout.decode("utf-8") if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode("utf-8") if isinstance(stderr, bytes) else stderr

        with open(stdout_file, "a+") as f:
            f.write(stdout)
        with open(stderr_file, "a+") as f:
            f.write(stderr)

        output = {}
        output["returncode"] = returncode

        # if os.path.exists('orca.gbw'):
        #    os.remove('orca.gbw')
        if os.path.exists("orca.densities"):
            os.remove("orca.densities")

        node_copy.get()

        if os.path.exists("orca_trj.xyz"):
            with open("out.txt", "r") as f:
                parsing = False
                found = False
                for line in f:
                    if re.search(
                        "FINAL ENERGY EVALUATION AT THE STATIONARY POINT", line
                    ):
                        parsing = True
                    if parsing:
                        if re.search("FINAL SINGLE POINT ENERGY", line):
                            parsing = False
                            found = True
            if not found:
                raise RuntimeError(
                    "OrcaCalculations fizzled! Not reaching stationary point."
                )

        if returncode != 0:
            raise RuntimeError(f"OrcaCalculations fizzled! Return code: {returncode}")

        return FWAction(stored_data=output)


class TurbomoleDefine(FireTaskBase):
    _fw_name = "TurbomoleDefine"

    def run_task(self, fw_spec):
        with open("coord", "w+") as coord:
            subprocess.call(["x2t", "mol.xyz"], stdout=coord)
        shutil.copy(fw_spec["tmol_template"], pathlib.Path.cwd() / "input.txt")

        self.CheckCoords()

        charge = fw_spec["charge"]
        if "ox" in fw_spec:
            charge = str(int(charge) + 1)
        elif "red" in fw_spec:
            charge = str(int(charge) - 1)

        if os.path.exists("control"):
            os.remove("control")

        stdout_err_file = pathlib.Path("define_out.txt").resolve()

        stdout = subprocess.PIPE
        stderr = subprocess.PIPE

        for line in fileinput.input("input.txt", inplace=1):
            line = line.replace("{{charge}}", charge)
            sys.stdout.write(line)

        with open("input.txt", "r") as define_in:
            p = subprocess.Popen(
                "define", stdout=stdout, stderr=stderr, stdin=define_in
            )
        (stdout, stderr) = p.communicate()
        returncode = p.returncode

        stdout = stdout.decode("utf-8") if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode("utf-8") if isinstance(stderr, bytes) else stderr

        with open(stdout_err_file, "a+") as f:
            f.write(stdout)
        with open(stdout_err_file, "a+") as f:
            f.write(stderr)

        output = {}
        output["returncode"] = returncode

        if returncode != 0:
            raise RuntimeError(f"Turbomole define fizzled! Return code: {returncode}")

        with open("define_out.txt", "r") as f:
            found = False
            for line in f:
                if re.search("define ended normally", line):
                    found = True
        if not found:
            raise RuntimeError("Turbomole define fizzled!")

        return FWAction(stored_data=output)

    def CheckCoords(self):
        with open("coord", "r") as f:
            coord_file = f.readlines()
        minima = {"x": 0, "y": 0, "z": 0}
        coords = {"x": 0, "y": 0, "z": 0}
        parsing = False
        for line in coord_file:
            if parsing:
                if re.match("\$", line):
                    parsing = False
                    break
                coords["x"], coords["y"], coords["z"], _ = line.split()
                for coord, minimum in minima.items():
                    minima[coord] = min(np.longdouble(coords[coord]), minimum)
            if re.match("\$coord", line):
                parsing = True
        coord_shift = {}
        for coord, minimum in minima.items():
            if minimum <= -100:
                coord_shift[coord] = -np.floor(minimum)
        if coord_shift:
            self.ShiftCoords(coord_file, coord_shift)

    def ShiftCoords(self, coord_file, coord_shift):
        parsing = False
        coords = {"x": 0, "y": 0, "z": 0}
        with open("coord", "w") as outfile:
            for line in coord_file:
                if parsing:
                    if re.match("\$", line):
                        parsing = False
                        outfile.write(line)
                        continue
                    coords["x"], coords["y"], coords["z"], atom = line.split()
                    for coord, shift in coord_shift.items():
                        coords[coord] = np.format_float_positional(
                            np.longdouble(coords[coord]) + shift, 14, unique=False
                        )
                    outfile.write(
                        f'{(20-len(str(coords["x"])))*" "}{coords["x"]}  {(20-len(str(coords["y"])))*" "}{coords["y"]}  {(20-len(str(coords["z"])))*" "}{coords["z"]}      {atom}\n'
                    )
                else:
                    outfile.write(line)
                    if re.match("\$coord", line):
                        parsing = True


class TurbomoleCosmoprep(FireTaskBase):
    _fw_name = "TurbomoleCosmoprep"

    def run_task(self, fw_spec):
        stdout_err_file = pathlib.Path("cosmoprep_out.txt").resolve()

        stdout = subprocess.PIPE
        stderr = subprocess.PIPE

        with open(fw_spec["tmol_cosmo_template"], "r") as cosmo_in:
            p = subprocess.Popen(
                "cosmoprep", stdout=stdout, stderr=stderr, stdin=cosmo_in
            )
        (stdout, stderr) = p.communicate()
        returncode = p.returncode

        stdout = stdout.decode("utf-8") if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode("utf-8") if isinstance(stderr, bytes) else stderr

        with open(stdout_err_file, "a+") as f:
            f.write(stdout)
        with open(stdout_err_file, "a+") as f:
            f.write(stderr)

        output = {}
        output["returncode"] = returncode

        if returncode != 0:
            raise RuntimeError(
                f"Turbomole cosmoprep fizzled! Return code: {returncode}"
            )

        with open("control", "a") as control:
            control.write("$cosmo_isorad\n")

        return FWAction(stored_data=output, update_spec={"cosmo": True})


class TurbomoleCalculations(FireTaskBase):
    _fw_name = "TurbomoleCalculations"

    def run_task(self, fw_spec):
        node_copy = NodeCopy()
        node_copy.put()

        stdout_file = pathlib.Path("out.txt").resolve()
        stderr_file = pathlib.Path("err.txt").resolve()

        stdout = subprocess.PIPE
        stderr = subprocess.PIPE

        p = subprocess.Popen("ridft", stdout=stdout, stderr=stderr)
        (stdout, stderr) = p.communicate()
        returncode = p.returncode

        stdout = stdout.decode("utf-8") if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode("utf-8") if isinstance(stderr, bytes) else stderr

        with open(stdout_file, "a+") as f:
            f.write(stdout)
        with open(stderr_file, "a+") as f:
            f.write(stderr)

        output = {}
        output["returncode"] = returncode

        if os.path.exists("mos"):
            os.remove("mos")
        if os.path.exists("alpha"):
            os.remove("alpha")
        if os.path.exists("beta"):
            os.remove("beta")
        if glob.glob("diff_*"):
            for junk_file in glob.glob("diff_*"):
                os.remove(junk_file)
        if glob.glob("diis_*"):
            for junk_file in glob.glob("diis_*"):
                os.remove(junk_file)
        if glob.glob("*.tmp"):
            for junk_file in glob.glob("*.tmp"):
                os.remove(junk_file)
        if glob.glob("MPI-TEMPDIR*"):
            for junk_folder in glob.glob("MPI-TEMPDIR*"):
                if os.path.isdir(junk_folder):
                    shutil.rmtree(junk_folder)

        node_copy.get()

        if os.path.exists("dscf_problem"):
            raise RuntimeError("TurbomoleCalculations fizzled during DSCF!")

        if "cosmo" not in fw_spec:
            with open("out.energy", "w+") as energy:
                subprocess.run("t2energy", stdout=energy, check=True)

        if returncode != 0:
            raise RuntimeError(
                f"TurbomoleCalculations fizzled! Return code: {returncode}"
            )

        return FWAction(stored_data=output)


class CosmothermCalculations(FireTaskBase):
    _fw_name = "CosmothermCalculations"

    def run_task(self, fw_spec):
        node_copy = NodeCopy()
        node_copy.put()

        if "ox" not in fw_spec and "red" not in fw_spec:
            input_files = [
                fw_spec["cosmotherm_gsolv_template"],
                fw_spec["cosmotherm_logp_template"],
            ]
        else:
            input_files = [fw_spec["cosmotherm_gsolv_template"]]

        stdout_file = pathlib.Path("out.txt").resolve()
        stderr_file = pathlib.Path("err.txt").resolve()
        returncodes = []

        for input_file in input_files:
            stdout = subprocess.PIPE
            stderr = subprocess.PIPE

            shutil.copy(input_file, pathlib.Path.cwd())

            command = "cosmotherm " + input_file

            p = subprocess.Popen(command, stdout=stdout, stderr=stderr, shell=True)
            (stdout, stderr) = p.communicate()
            returncodes.append(p.returncode)

            stdout = stdout.decode("utf-8") if isinstance(stdout, bytes) else stdout
            stderr = stderr.decode("utf-8") if isinstance(stderr, bytes) else stderr

            with open(stdout_file, "a+") as f:
                f.write(stdout)
            with open(stderr_file, "a+") as f:
                f.write(stderr)

            if p.returncode != 0:
                break

        output = {}
        output["returncode"] = returncodes[-1]

        node_copy.get()

        if sum(returncodes) != 0:
            raise RuntimeError(
                f"CosmothermCalculations fizzled! Return code: {returncodes}"
            )

        return FWAction(stored_data=output)


class XtbOut2OrcaIn(FireTaskBase):
    _fw_name = "XtbOut2OrcaIn"

    def run_task(self, fw_spec):
        shutil.move("mol.xyz", "mol_xtb.xyz")
        shutil.copy("xtbopt.xyz", "mol.xyz")


class OrcaOrbitalParser(FireTaskBase):
    _fw_name = "OrcaOrbitalParser"

    def run_task(self, fw_spec):
        with open("out.txt", "r") as outfile:
            if "red" in fw_spec:
                orbitals = []
                counter = 0
                parsing = False
                for line in outfile:
                    if re.search("\s*NO\s*OCC\s*E\(Eh\)\s*E\(eV\)\s*", line):
                        counter += 1
                        parsing = True
                    elif line.startswith("\n"):
                        if parsing and counter == 1:
                            orbitals_up = orbitals
                            orbitals = []
                        elif parsing and counter == 2:
                            orbitals_down = orbitals
                            orbitals = []
                        parsing = False
                    if parsing:
                        orbital = line.split()
                        orbitals.append(
                            {
                                "no": orbital[0],
                                "occupancy": orbital[1],
                                "eh": orbital[2],
                                "ev": orbital[3],
                            }
                        )
                del orbitals_up[0]  # remove off by one
                del orbitals_down[0]  # remove off by one

                homo = list(filter(lambda x: x["occupancy"] == "1.0000", orbitals_up))[
                    -1
                ]["ev"]

                lumo = list(
                    filter(lambda x: x["occupancy"] == "0.0000", orbitals_down)
                )[0]["ev"]

                homolumo = str(np.round(float(homo) - float(lumo), 4))

                return FWAction(
                    stored_data={
                        "HOMO_red": homo,
                        "LUMO_red": lumo,
                        "HOMO-LUMO-gap_red": homolumo,
                    }
                )

            else:
                orbitals = (
                    []
                )  # Modifed from Orca orbital parser by Dr. Juha Siitonen Rice University
                for line in outfile:
                    if re.search("\s*NO\s*OCC\s*E\(Eh\)\s*E\(eV\)\s*", line):
                        parsing = True
                    elif line.startswith("\n"):
                        parsing = False
                    if parsing:
                        orbital = line.split()
                        orbitals.append(
                            {
                                "no": orbital[0],
                                "occupancy": orbital[1],
                                "eh": orbital[2],
                                "ev": orbital[3],
                            }
                        )
                del orbitals[0]  # remove off by one

                homo = list(filter(lambda x: x["occupancy"] == "2.0000", orbitals))[-1][
                    "ev"
                ]

                lumo = list(filter(lambda x: x["occupancy"] == "0.0000", orbitals))[0][
                    "ev"
                ]

                homolumo = str(np.round(float(homo) - float(lumo), 4))

                return FWAction(
                    stored_data={"HOMO": homo, "LUMO": lumo, "HOMO-LUMO-gap": homolumo}
                )


class OrcaElectronicParser(FireTaskBase):
    _fw_name = "OrcaElectronicParser"

    def run_task(self, fw_spec):
        with open("out.txt", "r") as outfile:
            for line in outfile:
                if re.search("Magnitude \(Debye\)", line):
                    line = line.split()
                    line = line[3]
                    dipole = line
                if re.search("Isotropic polarizability", line):
                    line = line.split()
                    line = line[3]
                    polarizability = line
                if re.search("Isotropic quadrupole", line):
                    line = line.split()
                    line = line[3]
                    quadrupole = line
        if "red" in fw_spec:
            return FWAction(
                stored_data={
                    "dipole_moment_red": dipole,
                    "quadrupole_moment_red": quadrupole,
                    "polarizability_red": polarizability,
                }
            )
        elif "orca_origin" in fw_spec:
            return FWAction(
                stored_data={
                    "dipole_moment_N": dipole,
                    "quadrupole_moment_N": quadrupole,
                    "polarizability_N": polarizability,
                }
            )
        else:
            return FWAction(
                stored_data={
                    "dipole_moment": dipole,
                    "quadrupole_moment": quadrupole,
                    "polarizability": polarizability,
                }
            )


class OrcaFreqeuncyParser(FireTaskBase):
    _fw_name = "OrcaFrequencyParser"

    def run_task(self, fw_spec):
        with open("out.txt", "r") as outfile:
            parsing = False
            for line in outfile:
                if re.search("Final Gibbs free energy", line):
                    line = line.split()
                    line = line[5]
                    gibbs = line
                if re.search("IR SPECTRUM", line):
                    parsing = True
                    frequencies = []
                    intensities = []
                if parsing:
                    if re.match("\d", line.strip()):
                        entries = line.split()
                        frequency = entries[1]
                        frequencies.append(frequency)
                        intensity = entries[3]
                        intensities.append(intensity)
                    elif re.search("THERMOCHEMISTRY", line):
                        parsing = False
        if float(frequencies[6]) <= 0:
            freq_state = "negative"
        else:
            freq_state = "positive"
        if "ox" in fw_spec:
            return FWAction(
                stored_data={
                    "gibbs_ox": gibbs,
                    "freq_state_ox": freq_state,
                    "frequencies_ox": frequencies,
                    "intensities_ox": intensities,
                }
            )
        elif "red" in fw_spec:
            return FWAction(
                stored_data={
                    "gibbs_red": gibbs,
                    "freq_state_red": freq_state,
                    "frequencies_red": frequencies,
                    "intensities_red": intensities,
                }
            )
        else:
            return FWAction(
                stored_data={
                    "gibbs": gibbs,
                    "freq_state": freq_state,
                    "frequencies": frequencies,
                    "intensities": intensities,
                }
            )


class CosmothermCommonParser(FireTaskBase):
    _fw_name = "CosmothermCommonParser"

    def run_task(self, fw_spec):
        parsing = False
        with open("gsolv.out", "r") as outfile:
            for line in outfile:
                if re.search("(COSMO file ./mol.cosmo)", line):
                    parsing = True
                elif line.startswith("\n"):
                    parsing = False
                if parsing:
                    if re.search("Area", line):
                        area = line.split()
                        area = area[2]
                    elif re.search("Volume", line):
                        volume = line.split()
                        volume = volume[2]
                    elif re.search("Molecular Weight", line):
                        mw = line.split()
                        mw = mw[3]
                    elif re.search("accept", line):
                        h_acceptor = line.split()
                        h_acceptor = h_acceptor[4]
                    elif re.search("donor", line):
                        h_donor = line.split()
                        h_donor = h_donor[4]
        return FWAction(
            stored_data={
                "area": area,
                "volume": volume,
                "molecular_weight": mw,
                "h_acceptor_moment": h_acceptor,
                "h_donor_moment": h_donor,
            }
        )


class CosmothermLogpParser(FireTaskBase):
    _fw_name = "CosmothermLogpParser"

    def run_task(self, fw_spec):
        logp_list = []
        with open("logp.tab", "r") as outfile:
            for line in outfile:
                if re.search("3 mol", line):
                    logp = line.split()
                    logp_list.append(logp[2])
        return FWAction(
            stored_data={"logp_wet": logp_list[0], "logp_dry": logp_list[1]}
        )


class CosmothermGsolvParser(FireTaskBase):
    _fw_name = "CosmothermGsolvParser"

    def run_task(self, fw_spec):
        with open("gsolv.tab", "r") as outfile:
            for line in outfile:
                if re.search("2 mol", line):
                    gsolv = line.split()
                    gsolv = gsolv[-1]
        if "ox" in fw_spec:
            return FWAction(stored_data={"gsolv_ox": gsolv})
        elif "red" in fw_spec:
            return FWAction(stored_data={"gsolv_red": gsolv})
        else:
            return FWAction(stored_data={"gsolv": gsolv})


class OrcaEnergyParser(FireTaskBase):
    _fw_name = "OrcaEnergyParser"

    def run_task(self, fw_spec):
        if fw_spec.get("type") in ["E_red_neutral", "E_neutral_red"]:
            with open("out.txt", "r") as outfile:
                for line in outfile:
                    if re.search("FINAL SINGLE POINT ENERGY", line):
                        energy = line.split()
                        energy = energy[4]
            if "red" in fw_spec:
                return FWAction(stored_data={"energy_neutral_red": energy})
            else:
                return FWAction(stored_data={"energy_red_neutral": energy})
        elif fw_spec.get("type") in ["geo", "geo_red"]:
            with open("out.txt", "r") as outfile:
                parsing = False
                for line in outfile:
                    if re.search(
                        "FINAL ENERGY EVALUATION AT THE STATIONARY POINT", line
                    ):
                        parsing = True
                    if parsing:
                        if re.search("FINAL SINGLE POINT ENERGY", line):
                            energy = line.split()
                            energy = energy[4]
                            parsing = False
            if "red" in fw_spec:
                return FWAction(stored_data={"energy_neutral_neutral": energy})
            else:
                return FWAction(stored_data={"energy_red_red": energy})
        else:
            return


class CrestFragmentParser(FireTaskBase):
    _fw_name = "CrestFragmentParser"

    def run_task(self, fw_spec):
        with open("out.txt", "r") as outfile:
            for line in outfile:
                if re.search("# fragment in coord", line):
                    fragments = line.split()
                    fragments = int(fragments[5])
        if "ox" in fw_spec:
            return FWAction(stored_data={"fragments_ox": fragments})
        elif "red" in fw_spec:
            return FWAction(stored_data={"fragments_red": fragments})
        else:
            return FWAction(stored_data={"fragments": fragments})
