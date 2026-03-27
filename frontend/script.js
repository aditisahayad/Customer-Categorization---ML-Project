/**
 * script.js
 * =========
 * Frontend logic for the Customer Categorization prediction form.
 * Uses the Fetch API to call POST /predict and dynamically displays results.
 */

// --- DOM Elements ---
const form = document.getElementById("prediction-form");
const submitBtn = document.getElementById("submit-btn");
const resultSection = document.getElementById("result-section");
const errorSection = document.getElementById("error-section");
const errorMessage = document.getElementById("error-message");
const resultCluster = document.getElementById("result-cluster");
const resultCategory = document.getElementById("result-category");
const resultBadge = document.getElementById("result-badge");

// --- API Configuration ---
const API_URL = window.location.origin;

/**
 * Badge class mapping based on cluster value.
 */
const BADGE_MAP = {
    0: { cls: "low", text: "Low Value" },
    1: { cls: "medium", text: "Medium Value" },
    2: { cls: "high", text: "High Value" },
};

/**
 * Show the loading state on the submit button.
 */
function setLoading(isLoading) {
    if (isLoading) {
        submitBtn.classList.add("loading");
        submitBtn.disabled = true;
    } else {
        submitBtn.classList.remove("loading");
        submitBtn.disabled = false;
    }
}

/**
 * Display the prediction result with animation.
 */
function showResult(cluster, category) {
    // Hide error
    errorSection.classList.add("hidden");

    // Update result values
    resultCluster.textContent = cluster;
    resultCategory.textContent = category;

    // Set badge
    const badge = BADGE_MAP[cluster] || { cls: "medium", text: "Unknown" };
    resultBadge.textContent = badge.text;
    resultBadge.className = "result-badge " + badge.cls;

    // Show result section with animation
    resultSection.classList.remove("hidden");
    resultSection.style.animation = "none";
    // Trigger reflow to restart animation
    void resultSection.offsetWidth;
    resultSection.style.animation = "fadeSlideUp 0.5s ease-out";

    // Scroll to result
    resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/**
 * Display an error message.
 */
function showError(message) {
    resultSection.classList.add("hidden");
    errorMessage.textContent = message;
    errorSection.classList.remove("hidden");
    errorSection.style.animation = "none";
    void errorSection.offsetWidth;
    errorSection.style.animation = "shake 0.4s ease-out";
}

/**
 * Validate form inputs before submission.
 * Returns an object with the validated data or null if invalid.
 */
function validateForm() {
    const age = parseInt(document.getElementById("input-age").value, 10);
    const income = parseFloat(document.getElementById("input-income").value);
    const spending = parseFloat(document.getElementById("input-spending").value);
    const children = parseInt(document.getElementById("input-children").value, 10);
    const education = document.getElementById("input-education").value;

    if (isNaN(age) || age < 10 || age > 120) {
        showError("Please enter a valid age between 10 and 120.");
        return null;
    }
    if (isNaN(income) || income < 0) {
        showError("Please enter a valid income (0 or greater).");
        return null;
    }
    if (isNaN(spending) || spending < 0) {
        showError("Please enter a valid total spending (0 or greater).");
        return null;
    }
    if (isNaN(children) || children < 0 || children > 10) {
        showError("Please enter a valid number of children (0–10).");
        return null;
    }
    if (education === "" || education === null) {
        showError("Please select an education level.");
        return null;
    }

    return {
        Age: age,
        Income: income,
        Total_Spending: spending,
        Children: children,
        Education: parseInt(education, 10),
    };
}

/**
 * Handle form submission — call the /predict endpoint.
 */
async function handleSubmit(event) {
    event.preventDefault();

    // Validate
    const data = validateForm();
    if (!data) return;

    setLoading(true);

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });

        if (!response.ok) {
            const errBody = await response.json().catch(() => ({}));
            const detail = errBody.detail || `Server error (${response.status})`;
            throw new Error(detail);
        }

        const result = await response.json();
        showResult(result.cluster, result.category);
    } catch (err) {
        showError(err.message || "Something went wrong. Please try again.");
    } finally {
        setLoading(false);
    }
}

// --- Event Listeners ---
form.addEventListener("submit", handleSubmit);
