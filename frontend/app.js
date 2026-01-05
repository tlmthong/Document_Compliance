// API Configuration
const PROCESS_FLOW_URL = 'http://127.0.0.1:6868';
const JUDGE_FLOW_URL = 'http://127.0.0.1:6969';

// Initialize field counter
let fieldCounter = 0;

// DOM Elements
document.addEventListener('DOMContentLoaded', function() {
    // Add initial field
    addField();
    
    // Setup form handlers
    document.getElementById('upload-form').addEventListener('submit', handleUpload);
    document.getElementById('judge-form').addEventListener('submit', handleJudge);
});

// Menu Navigation
function showMenu(menu) {
    // Update nav buttons
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    // Update menu sections
    document.querySelectorAll('.menu-section').forEach(section => section.classList.remove('active'));
    
    if (menu === 'upload') {
        document.getElementById('upload-menu').classList.add('active');
    } else {
        document.getElementById('judge-menu').classList.add('active');
    }
}

// Upload Policy Handler
async function handleUpload(e) {
    e.preventDefault();
    
    const policyId = document.getElementById('policy-id').value;
    const subject = document.getElementById('subject').value;
    const fileInput = document.getElementById('policy-file');
    const file = fileInput.files[0];
    
    if (!file) {
        showUploadResult('Please select a file', true);
        return;
    }
    
    showLoading(true);
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const url = `${PROCESS_FLOW_URL}/upload_policy?policy_id=${encodeURIComponent(policyId)}&subject=${encodeURIComponent(subject)}`;
        
        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            showUploadResult(`✓ Policy "${policyId}" uploaded successfully!`, false);
        } else {
            const result = await response.json();
            showUploadResult(`✗ Upload failed: ${result.detail || JSON.stringify(result)}`, true);
        }
    } catch (error) {
        showUploadResult(`✗ Error: ${error.message}`, true);
    } finally {
        showLoading(false);
    }
}

// Judge Handler
async function handleJudge(e) {
    e.preventDefault();
    
    const policyId = document.getElementById('judge-policy-id').value;
    const customerJson = buildCustomerJson();
    
    showLoading(true);
    
    try {
        const url = `${JUDGE_FLOW_URL}/judge?policy_id=${encodeURIComponent(policyId)}`;
        
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(customerJson)
        });
        
        const result = await response.json();
        const resultBox = document.getElementById('judge-result');
        
        resultBox.classList.remove('hidden', 'success', 'error');
        
        if (response.ok) {
            resultBox.classList.add('success');
            displayJudgeResult(result);
        } else {
            resultBox.classList.add('error');
            displayJudgeError(result.detail || JSON.stringify(result));
        }
    } catch (error) {
        const resultBox = document.getElementById('judge-result');
        resultBox.classList.remove('hidden', 'success');
        resultBox.classList.add('error');
        displayJudgeError(error.message);
    } finally {
        showLoading(false);
    }
}

// Display Judge Result
function displayJudgeResult(result) {
    const decisionBox = document.getElementById('final-decision');
    const reasonsBox = document.getElementById('reasons-list');
    
    // Display Final Decision
    const finalDecision = result['Final Decision'] || result['final_decision'] || 'N/A';
    const decisionClass = finalDecision.toLowerCase().includes('approve') || finalDecision.toLowerCase().includes('pass') ? 'approved' : 'rejected';
    
    decisionBox.innerHTML = `
        <h3>Final Decision</h3>
        <div class="decision-value ${decisionClass}">${finalDecision}</div>
    `;
    
    // Display Reasons
    const reasons = result['Reasons'] || result['reasons'] || [];
    
    let reasonsHtml = '<h3>Reasons</h3><ul class="reasons-ul">';
    if (Array.isArray(reasons) && reasons.length > 0) {
        reasons.forEach((reason, index) => {
            reasonsHtml += `<li><span class="reason-num">${index + 1}.</span> ${reason}</li>`;
        });
    } else {
        reasonsHtml += '<li>No reasons provided</li>';
    }
    reasonsHtml += '</ul>';
    
    reasonsBox.innerHTML = reasonsHtml;
}

// Display Judge Error
function displayJudgeError(errorMsg) {
    const decisionBox = document.getElementById('final-decision');
    const reasonsBox = document.getElementById('reasons-list');
    
    decisionBox.innerHTML = `
        <h3>Error</h3>
        <div class="decision-value rejected">${errorMsg}</div>
    `;
    reasonsBox.innerHTML = '';
}

// Show Upload Result
function showUploadResult(message, isError) {
    const resultBox = document.getElementById('upload-result');
    resultBox.classList.remove('hidden', 'success', 'error');
    resultBox.classList.add(isError ? 'error' : 'success');
    resultBox.innerHTML = `<div class="upload-message ${isError ? 'error-msg' : 'success-msg'}">${message}</div>`;
}

// Add Customer Field
function addField() {
    const fieldsContainer = document.getElementById('customer-fields');
    const fieldId = fieldCounter++;
    
    const fieldRow = document.createElement('div');
    fieldRow.className = 'field-row';
    fieldRow.id = `field-row-${fieldId}`;
    
    fieldRow.innerHTML = `
        <input type="text" class="field-key" placeholder="Field name" onchange="updateJsonPreview()" oninput="updateJsonPreview()">
        <input type="text" class="field-value" placeholder="Field value" onchange="updateJsonPreview()" oninput="updateJsonPreview()">
        <button type="button" class="delete-field-btn" onclick="deleteField(${fieldId})">×</button>
    `;
    
    fieldsContainer.appendChild(fieldRow);
    updateJsonPreview();
}

// Delete Customer Field
function deleteField(fieldId) {
    const fieldRow = document.getElementById(`field-row-${fieldId}`);
    if (fieldRow) {
        fieldRow.remove();
        updateJsonPreview();
    }
}

// Build Customer JSON from fields
function buildCustomerJson() {
    const json = {};
    const fieldRows = document.querySelectorAll('#customer-fields .field-row');
    
    fieldRows.forEach(row => {
        const keyInput = row.querySelector('.field-key');
        const valueInput = row.querySelector('.field-value');
        
        if (keyInput && valueInput && keyInput.value.trim()) {
            const key = keyInput.value.trim();
            let value = valueInput.value;
            
            // Try to parse as number or boolean
            if (value === 'true') {
                value = true;
            } else if (value === 'false') {
                value = false;
            } else if (!isNaN(value) && value.trim() !== '') {
                value = parseFloat(value);
            }
            
            json[key] = value;
        }
    });
    
    return json;
}

// Update JSON Preview
function updateJsonPreview() {
    const json = buildCustomerJson();
    const preview = document.getElementById('json-preview');
    preview.textContent = JSON.stringify(json, null, 2);
}

// Show Result
function showResult(elementId, message, isError) {
    const resultBox = document.getElementById(elementId);
    resultBox.classList.remove('hidden', 'success', 'error');
    resultBox.classList.add(isError ? 'error' : 'success');
    resultBox.innerHTML = `<pre>${message}</pre>`;
}

// Show/Hide Loading
function showLoading(show) {
    const loading = document.getElementById('loading');
    if (show) {
        loading.classList.remove('hidden');
    } else {
        loading.classList.add('hidden');
    }
}
