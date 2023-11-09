from pathlib import Path

work_dir = Path("/home/shius/projects/xprize/_infer_original/_log_thre0/")
enc_file = work_dir / "plants_family-encoding.txt"
fam_inf  = work_dir / "fam_inference_encoding.txt"
out_file = work_dir / "fam_inference_encoding_with_labels.txt"

with open(enc_file, "r") as f:
  # {encoding:taxon_label}
  edict = {l.strip().split(" ")[1]:l.split(" ")[0] for l in f.readlines()}
  print(edict)

with open(fam_inf, "r") as f:
  finf_enc_list = [l.strip() for l in f.readlines()]

with open(out_file, "w") as f:
  for e in finf_enc_list:
    f.write(f"{e},{edict[e]}\n")
