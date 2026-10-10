
const form = document.getElementById("solverForm");
const statusBox = document.getElementById("status");
const solveButton = document.getElementById("solveButton");
const downloadButton = document.getElementById("downloadButton");

const summary = document.getElementById("summary");
const graphCard = document.getElementById("graphCard");
const errorCard = document.getElementById("errorCard");
const resultsCard = document.getElementById("resultsCard");

let chart = null;
let lastData = null;

function makeCell(row, value, heading = false) {
    const cell = document.createElement(heading ? "th" : "td");
    cell.textContent = value;
    row.appendChild(cell);
}

function formatNumber(value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) {
        return "—";
    }
    return Number(value).toPrecision(9);
}

function makeTable(headers, rows) {
    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");

    headers.forEach(header => makeCell(headerRow, header, true));
    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");

    rows.forEach(values => {
        const row = document.createElement("tr");
        values.forEach(value => makeCell(row, value));
        tbody.appendChild(row);
    });

    table.appendChild(tbody);
    return table;
}

function renderSummary(data) {
    summary.replaceChildren();

    const items = [
        ["Equation", `dy/dx = ${data.equation}`],
        ["Methods returned", Object.keys(data.results).length],
        ["Exact solution", data.exact_available ? data.exact_expression : "Not found"]
    ];

    items.forEach(([label, value]) => {
        const card = document.createElement("div");
        card.className = "summary-item";

        const small = document.createElement("span");
        small.textContent = label;

        const strong = document.createElement("strong");
        strong.textContent = String(value);

        card.append(small, strong);
        summary.appendChild(card);
    });

    summary.hidden = false;
}

function renderGraph(data) {
    const canvas = document.getElementById("solutionChart");

    if (chart) {
        chart.destroy();
        chart = null;
    }

    const datasets = [];
    const palette = [
        "#2563eb", "#dc2626", "#059669", "#9333ea",
        "#d97706", "#0891b2", "#4f46e5", "#be123c",
        "#0f766e", "#7c3aed", "#a16207", "#475569",
        "#db2777"
    ];

    let colorIndex = 0;

    for (const [key, result] of Object.entries(data.results)) {
        if (!result.available || !result.rows.length) continue;

        const color = palette[colorIndex++ % palette.length];

        datasets.push({
            label: result.label,
            data: result.rows.map(row => ({ x: row.x, y: row.y })),
            borderColor: color,
            backgroundColor: color,
            borderWidth: 2,
            pointRadius: result.rows.length > 100 ? 0 : 2,
            tension: 0,
            showLine: true
        });
    }

    // Exact values are already included in each result row when available.
    // Draw one separate exact-solution curve, using the first available row set.
    const firstResult = Object.values(data.results)
        .find(result => result.available && result.rows.length &&
            result.rows[0].exact !== undefined);

    if (firstResult) {
        datasets.push({
            label: "Exact analytical solution",
            data: firstResult.rows.map(row => ({
                x: row.x, y: row.exact
            })),
            borderColor: "#111827",
            borderWidth: 3,
            borderDash: [7, 5],
            pointRadius: 0,
            showLine: true
        });
    }

    chart = new Chart(canvas, {
        type: "scatter",
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: "nearest",
                intersect: false
            },
            scales: {
                x: {
                    type: "linear",
                    title: { display: true, text: "x" }
                },
                y: {
                    title: { display: true, text: "y(x)" }
                }
            },
            plugins: {
                legend: { position: "bottom" },
                tooltip: { enabled: true }
            }
        }
    });

    graphCard.hidden = datasets.length === 0;
}

function renderErrors(data) {
    const container = document.getElementById("errorTable");
    container.replaceChildren();

    if (!data.exact_available) {
        const note = document.createElement("p");
        note.textContent =
            "An exact solution was not found. Numerical errors against an exact solution cannot be calculated.";
        container.appendChild(note);
        errorCard.hidden = false;
        return;
    }

    const rows = [];

    for (const [key, error] of Object.entries(data.errors)) {
        rows.push([
            data.results[key]?.label || key,
            formatNumber(error.final_abs_error),
            formatNumber(error.max_abs_error)
        ]);
    }

    container.appendChild(makeTable(
        ["Method", "Final absolute error", "Maximum absolute error"],
        rows
    ));

    errorCard.hidden = false;
}

function renderResults(data) {
    const container = document.getElementById("resultsTables");
    container.replaceChildren();

    document.getElementById("resultDescription").textContent =
        data.exact_available
            ? `Exact solution found: ${data.exact_expression}`
            : "No exact solution was found automatically. Showing numerical results where available.";

    for (const [key, result] of Object.entries(data.results)) {
        const heading = document.createElement("h3");
        heading.className = "method-heading";
        heading.textContent = result.label;
        container.appendChild(heading);

        if (!result.available) {
            const message = document.createElement("p");
            message.textContent = result.message || "No result available.";
            container.appendChild(message);
            continue;
        }

        const rows = result.rows.map(row => {
            const values = [
                row.step,
                formatNumber(row.x),
                formatNumber(row.y)
            ];

            if (row.exact !== undefined) {
                values.push(formatNumber(row.exact));
                values.push(formatNumber(row.abs_error));
            }

            return values;
        });

        const headers = ["Step", "x", "Numerical y"];
        if (data.exact_available) {
            headers.push("Exact y", "Absolute error");
        }

        const wrapper = document.createElement("div");
        wrapper.className = "table-wrap";
        wrapper.appendChild(makeTable(headers, rows));
        container.appendChild(wrapper);
    }

    resultsCard.hidden = false;
}

form.addEventListener("submit", async event => {
    event.preventDefault();

    solveButton.disabled = true;
    downloadButton.disabled = true;
    statusBox.className = "";
    statusBox.textContent = "Calculating numerical solution...";

    summary.hidden = true;
    graphCard.hidden = true;
    errorCard.hidden = true;
    resultsCard.hidden = true;

    if (chart) {
        chart.destroy();
        chart = null;
    }

  const payload = {
    equation: document.getElementById("equation").value.trim(),
    method: document.getElementById("method").value,
    x0: document.getElementById("x0").value,
    y0: document.getElementById("y0").value,
    h: document.getElementById("h").value,
    end: document.getElementById("end").value
};

    try {
        const response = await fetch("/solve", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Calculation failed.");
        }

        lastData = data;
        renderSummary(data);
        renderGraph(data);
        renderErrors(data);
        renderResults(data);

        statusBox.className = "success";
        statusBox.textContent = "Calculation completed successfully.";
        downloadButton.disabled = false;
    } catch (error) {
        lastData = null;
        statusBox.className = "error";
        statusBox.textContent = error.message;
    } finally {
        solveButton.disabled = false;
    }
});

downloadButton.addEventListener("click", () => {
    if (!lastData) return;

    const csvRows = [["Method", "Step", "x", "Numerical y",
                      "Exact y", "Absolute error"]];

    for (const [key, result] of Object.entries(lastData.results)) {
        if (!result.available) continue;

        result.rows.forEach(row => {
            csvRows.push([
                result.label,
                row.step,
                row.x,
                row.y,
                row.exact ?? "",
                row.abs_error ?? ""
            ]);
        });
    }

    // Escape quotes and wrap every field for valid CSV output.
    const csv = csvRows.map(row =>
        row.map(value =>
            `"${String(value).replaceAll('"', '""')}"`
        ).join(",")
    ).join("\r\n");

    const blob = new Blob(["\uFEFF" + csv], {
        type: "text/csv;charset=utf-8;"
    });

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "numerical-methods-results.csv";
    link.click();
    URL.revokeObjectURL(url);
});
