import joblib

def export_tags_to_files(encoders_path='encoders.pkl'):
    encoders = joblib.load(encoders_path)

    data_structure = {}
    for key, encoder in encoders.items():
        if hasattr(encoder, 'classes_'):
            data_structure[key] = encoder.classes_.tolist()

    txt_filename = 'tags_list.txt'
    with open(txt_filename, 'w', encoding='utf-8') as f:
        for category, tags in data_structure.items():
            f.write(f"--- Category: {category.upper()} ({len(tags)} tags) ---\n")
            for tag in tags:
                f.write(f"- {tag}\n")
            f.write("\n")

    print(f"Saved to file: {txt_filename}")

if __name__ == "__main__":
    export_tags_to_files()