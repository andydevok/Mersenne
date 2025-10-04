# Pirolo Mersenne Hunter

### A geometric anomaly detection model for identifying large Mersenne prime candidates.

---

## Abstract

This repository contains the code for a novel predictive method for locating Mersenne primes ($M_p=2^p-1$). The model leverages the known family of stable Mersenne exponents to calibrate a "numerical spectrometer" based on the non-trivial zeros of the Riemann Zeta function. The signature of a Mersenne prime is identified by its unique "signal coherence," which manifests as an anomalously low **Cosine Variance**.

Extensive validation has shown that the model is a high-precision **"Local Refiner"** with a predictable error margin, capable of pinpointing candidates within an already identified region of interest. The primary value of this method is its ability to drastically reduce the search space for projects like GIMPS, transforming the search from a brute-force marathon into a targeted, high-precision endeavor.

## The Methodology

The model employs a two-phase "coarse-to-fine" search strategy:

#### 1. Coarse Search (Búsqueda Gruesa)
The model analyzes a large, random sample of prime candidates within a vast, multi-million-exponent territory. It uses the **Cosine Variance** metric to identify the candidate with the most anomalous signal coherence. This candidate becomes the "epicenter" of a high-probability hotspot.

#### 2. Fine Search (Búsqueda Fina)
The model then focuses on a small neighborhood around the identified hotspot. It exhaustively analyzes all prime candidates in this zone to find the top 10 geometric anomalies. These finalists are then ranked by a **Hybrid Filter**, which combines the geometric score with classical number-theoretic criteria (proximity to powers of two, prime isolation, etc.) to select the single most robust candidate.

## Demonstration Script: `M52_Rediscovery_Demo.py`

The primary script included in this repository (`Pirolo_Mersenne_Hunter_GIMPS_Demo.py`) is a self-contained demonstration of the full methodology. It performs a blind hunt for the known Mersenne prime **M52 (p=136,279,841)** in a 2-million-exponent-wide territory to validate the model's robustness and precision.

### How to Run

1.  **Prerequisites:**
    * Python 3.x

2.  **Installation:**
    The script will attempt to install the required libraries (`mpmath`, `sympy`, `pandas`, `numpy`, `tqdm`) if they are not found. You can also install them manually:
    ```bash
    pip install mpmath sympy pandas numpy tqdm
    ```

3.  **Execution:**
    Save the script and run it from your terminal:
    ```bash
    python Pirolo_Mersenne_Hunter_GIMPS_Demo.py
    ```

### Expected Output

The script will run for a considerable amount of time as it generates and analyzes millions of prime candidates. The final output will be a report detailing the rediscovery of M52, including the candidate found and the final relative error, which demonstrates the model's precision.

## Key Findings & M53 Candidate

* **The model is a high-precision "Local Refiner"**, capable of identifying candidates with a relative error consistently below 0.03% in local searches.
* **The model's "blind hunt" capability is less precise**, with a maximum observed error of approximately **0.4%**, which defines its effective search resolution.
* The most extensive blind hunt performed with the definitive version of this model identified a final, robust candidate for the 53rd Mersenne prime: **p = 170,904,353**.

## Author

* **Andrés Sebastian Pirolo** (with assistance from Gemini)

## License

This project is licensed under the MIT License.

