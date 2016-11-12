def write_model_conf_to_file(model_summary,timestamp, path_to_file):
    out_to_file = open(path_to_file, "a")
    out_to_file.write("Reference to image: "+ timestamp + ".png"+ "\n")
    out_to_file.write(model_summary + "\n")
    out_to_file.close()