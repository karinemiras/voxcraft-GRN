import re

def extract_seconds_and_average(log_text: str):
    # Find all numbers followed by 's'
    seconds = [float(x) for x in re.findall(r'([\d.]+)s', log_text)]

    if not seconds:
        raise ValueError("No timing values found.")

    avg = sum(seconds) / len(seconds)

    return {
        "count": len(seconds),
        "average": avg,
        
        "min": min(seconds),
        "max": max(seconds),
    }


if __name__ == "__main__":
    for i in range (1,6):
        # Option 1: paste log directly
        with open("../working_data/voxlocbiggeno/crosspropxyv4s25_"+str(i)+".log", "r") as f:
            log_data = f.read()

        stats = extract_seconds_and_average(log_data)

        print(f"Samples : {stats['count']}")
        print(f"       Average : {stats['average']:.2f}s")
        print(f"Min     : {stats['min']:.2f}s")
        print(f"Max     : {stats['max']:.2f}s")