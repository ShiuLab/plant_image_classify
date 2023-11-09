'''
Shinhan Shiu
6/5/23
Check the performance of effnet models in model selection run
'''

from pathlib import Path


def get_test_acc(line):
  counts  = line.split(" (")[:-1][:-1].split("/")
  llist   = [token for token in line.split(" ") if token != '']
  label   = llist[3][:-1]
  n_acc   = int(counts[0])
  n_total = int(counts[1])

  return label, n_acc, n_total


model_dir   = Path("/mnt/research/xprize23/plants_test/family/model_selection")
model_files = model_dir.iterdir()
out_file    = model_dir / "log_effnet_model_performance.txt"

# {model: [epoch_dict, <-- {epoch:[acc_train, acc_valid]}, 
#          acc_test,   <-- {class_label:[acc, acc, n_acc, n_total]}
#          overall_n_acc, overall_n_total ]}
model_info = {}
for model_file in model_files:
  if str(model_file).find("log_family_") == -1:
    continue

  model = str(model_file).strip().split("log_family_tf_")[-1]
  print(model)

  epoch = acc_train = acc_valid = ""
  epoch_dict = {}
  acc_test = {} 
  overall_n_acc = overall_n_total = 0
  with open(model_file) as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
      if line.find("Epoch ") != -1:
        epoch = line.split(" ")[-1].split("/")[0]
        acc_train = lines[idx+2].split("Acc: ")[-1]
        acc_valid = lines[idx+3].split("Acc: ")[-1]
        print(f"  epoch:{epoch}, {acc_train}, {acc_valid}")

        epoch_dict[epoch] = [acc_train, acc_valid]
      
      elif line.find("Test Accuracy of ") != -1:
        label, n_acc, n_total = get_test_acc(line)
        print(f"  class={label}, {n_acc}, {n_total}")

        acc_test[label] = [n_acc, n_total]

      elif line.find("Test Accuracy (Overall):") != -1:
        counts = line.split(" (")[-1][:-1].split("/")
        overall = [int(counts[0]), int(counts[1])]
        print(f'  overall:{overall}')

  model_info[model] = [epoch_dict, acc_test, overall]  
  break

print(model_info)

with open(out_file, "w") as out:
  # header
  elist = [f"epoch_{idx}" for idx in range(10)]
  out_file.write(f'model\t{",".join(elist)}')

      

  


          
