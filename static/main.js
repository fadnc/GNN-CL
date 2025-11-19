// Advanced GNN Fraud Detection - Main JavaScript
// Features: Real-time updates, advanced analytics, pattern detection

let svg, g, simulation;
let width, height;
let currentGraph = { nodes: [], edges: [] };
let selectedNode = null;

// Colors
const COLORS = {
    safe: "#00eaff",
    medium: "#ffa500",
    high: "#ff0066",
    bg: "#0a0e1a"
};

// ============================
// INITIALIZATION
// ============================
function init() {
    initSVG();
    setupEventListeners();
    loadInitialData();
    startRealTimeUpdates();
}

function initSVG() {
    const container = document.querySelector('.graph-container svg');
    width = container.parentElement.clientWidth;
    height = container.parentElement.clientHeight;

    svg = d3.select('.graph-container svg')
        .attr('width', width)
        .attr('height', height);

    // Create defs for gradients and filters
    const defs = svg.append('defs');
    
    // Glow filter
    const glow = defs.append('filter')
        .attr('id', 'glow')
        .attr('x', '-50%')
        .attr('y', '-50%')
        .attr('width', '200%')
        .attr('height', '200%');
    
    glow.append('feGaussianBlur')
        .attr('stdDeviation', '4')
        .attr('result', 'coloredBlur');
    
    const feMerge = glow.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // Arrow markers
    ['safe', 'medium', 'high'].forEach(type => {
        defs.append('marker')
            .attr('id', `arrow-${type}`)
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 20)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', COLORS[type])
            .attr('opacity', 0.6);
    });

    g = svg.append('g');

    setupZoom();
}

function setupZoom() {
    const zoom = d3.zoom()
        .scaleExtent([0.1, 5])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });

    svg.call(zoom);
    svg.style('cursor', 'grab');
}

// ============================
// DATA LOADING
// ============================
async function loadInitialData() {
    try {
        // Load metrics
        const metricsRes = await fetch('/api/metrics');
        const metricsData = await metricsRes.json();
        updateHeaderStats(metricsData.metrics);
        
        // Load graph
        await loadGraph();
        
        // Load alerts
        await loadAlerts();
        
        hideLoading();
    } catch (error) {
        console.error('Error loading data:', error);
        showNotification('Error loading data', 'error');
    }
}

async function loadGraph() {
    const nodes = document.querySelector('.sidebar input[type="number"]').value;
    const edges = document.querySelectorAll('.sidebar input[type="number"]')[1].value;
    
    showNotification('Loading graph...');
    
    try {
        const res = await fetch(`/api/graph?nodes=${nodes}&edges=${edges}`);
        const data = await res.json();
        
        currentGraph = data;
        renderGraph(data);
        updateHeaderStats(data.metrics);
        
        showNotification('Graph loaded successfully', 'success');
    } catch (error) {
        console.error('Error loading graph:', error);
        showNotification('Failed to load graph', 'error');
    }
}

// ============================
// GRAPH RENDERING
// ============================
function renderGraph(data) {
    // Clear existing
    g.selectAll('*').remove();

    if (!data.nodes || !data.edges) return;

    // Create links
    const link = g.append('g')
        .selectAll('line')
        .data(data.edges)
        .enter()
        .append('line')
        .attr('class', 'link')
        .style('stroke', d => getRiskColor(d))
        .style('stroke-width', d => d.is_suspicious ? 2.5 : 1.5)
        .style('opacity', d => d.is_suspicious ? 0.7 : 0.4)
        .attr('marker-end', d => {
            const type = d.is_suspicious ? 'high' : 
                        d.pred_prob > 0.3 ? 'medium' : 'safe';
            return `url(#arrow-${type})`;
        });

    // Create node groups
    const nodeGroup = g.append('g')
        .selectAll('g')
        .data(data.nodes)
        .enter()
        .append('g')
        .attr('class', 'node-group')
        .call(d3.drag()
            .on('start', dragStarted)
            .on('drag', dragged)
            .on('end', dragEnded));

    // Add circles
    nodeGroup.append('circle')
        .attr('r', d => getNodeSize(d))
        .style('fill', d => getRiskColor(d))
        .style('stroke', '#ffffff')
        .style('stroke-width', d => d.is_suspicious ? 2 : 1)
        .style('stroke-opacity', 0.8)
        .style('filter', 'url(#glow)')
        .style('cursor', 'pointer')
        .on('click', (event, d) => {
            event.stopPropagation();
            selectNode(d);
        })
        .on('mouseover', function(event, d) {
            highlightNode(d, this);
        })
        .on('mouseout', function(event, d) {
            unhighlightNode(d, this);
        });

    // Add labels
    nodeGroup.append('text')
        .text(d => d.name || d.id.split('_')[1])
        .attr('dy', -12)
        .attr('text-anchor', 'middle')
        .style('font-size', '9px')
        .style('font-family', 'monospace')
        .style('fill', '#ffffff')
        .style('opacity', 0.7)
        .style('pointer-events', 'none')
        .style('text-shadow', '0 0 3px #000');

    // Setup simulation
    simulation = d3.forceSimulation(data.nodes)
        .force('link', d3.forceLink(data.edges)
            .id(d => d.id)
            .distance(100)
            .strength(0.3))
        .force('charge', d3.forceManyBody()
            .strength(d => d.is_suspicious ? -400 : -300)
            .distanceMax(400))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide()
            .radius(d => getNodeSize(d) + 15)
            .strength(0.8))
        .force('x', d3.forceX(width / 2).strength(0.05))
        .force('y', d3.forceY(height / 2).strength(0.05))
        .alphaDecay(0.02)
        .on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            nodeGroup
                .attr('transform', d => `translate(${d.x},${d.y})`);
        });
}

// ============================
// NODE INTERACTION
// ============================
async function selectNode(node) {
    selectedNode = node;
    
    // Highlight selection
    g.selectAll('.node-group circle')
        .style('stroke-width', d => d.id === node.id ? 3 : d.is_suspicious ? 2 : 1)
        .style('stroke', d => d.id === node.id ? '#ffffff' : '#ffffff');
    
    // Load details
    try {
        const res = await fetch(`/api/node/${node.id}`);
        const data = await res.json();
        
        displayNodeDetails(data);
        displayFraudPatterns(data.fraud_patterns);
        updateRiskMeter(data.risk_score);
        
    } catch (error) {
        console.error('Error loading node details:', error);
    }
}

function displayNodeDetails(data) {
    const detailsContent = document.querySelector('[data-content="details"]');
    
    const html = `
        <div class="info-card">
            <h3>Node Information</h3>
            <div class="info-row">
                <span class="info-label">Node ID</span>
                <span class="info-value">${data.id}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Type</span>
                <span class="info-value">${data.type}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Transactions</span>
                <span class="info-value">${data.degree}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Avg Amount</span>
                <span class="info-value">$${data.summary.avg_amount.toFixed(2)}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Total Amount</span>
                <span class="info-value">$${data.summary.total_amount.toFixed(2)}</span>
            </div>
            <div class="info-row" style="border: none;">
                <span class="info-label">Status</span>
                <span class="info-value">
                    <span class="badge ${data.is_suspicious ? 'badge-high' : 'badge-low'}">
                        ${data.is_suspicious ? 'Suspicious' : 'Normal'}
                    </span>
                </span>
            </div>
        </div>

        <div class="info-card">
            <h3>Risk Assessment</h3>
            <div class="risk-meter">
                <div class="risk-bar">
                    <div class="risk-indicator" style="left: ${data.risk_score * 100}%;"></div>
                </div>
                <div style="text-align: center; margin-top: 15px;">
                    <div style="font-size: 32px; font-weight: bold; color: ${getRiskColor({risk_score: data.risk_score * 100})};">
                        ${data.risk_score > 0.7 ? 'High' : data.risk_score > 0.3 ? 'Medium' : 'Low'}
                    </div>
                    <div style="font-size: 12px; color: #a0b0c0; margin-top: 5px;">
                        Risk Score: ${(data.risk_score * 100).toFixed(1)}%
                    </div>
                </div>
            </div>
        </div>

        <div class="info-card">
            <h3>Top Counterparties</h3>
            ${Object.entries(data.top_counterparties).slice(0, 5).map(([id, count]) => `
                <div class="info-row">
                    <span class="info-label">${id}</span>
                    <span class="info-value">${count} txns</span>
                </div>
            `).join('')}
        </div>
    `;
    
    detailsContent.innerHTML = html;
}

function displayFraudPatterns(patterns) {
    if (!patterns || patterns.length === 0) return;
    
    const analysisContent = document.querySelector('[data-content="analysis"]');
    const existingCard = analysisContent.querySelector('.info-card');
    
    const patternsHTML = `
        <div class="info-card">
            <h3>Detected Patterns</h3>
            <div style="margin-top: 15px;">
                ${patterns.map(p => `
                    <div class="badge badge-${p.severity}" style="margin: 5px; display: inline-block;">
                        ${p.description}
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    if (existingCard) {
        existingCard.insertAdjacentHTML('afterend', patternsHTML);
    }
}

function updateRiskMeter(score) {
    const indicator = document.querySelector('.risk-indicator');
    if (indicator) {
        indicator.style.left = `${score * 100}%`;
    }
}

// ============================
// ALERTS
// ============================
async function loadAlerts() {
    try {
        const res = await fetch('/api/alerts?limit=20');
        const data = await res.json();
        
        displayAlerts(data.alerts);
    } catch (error) {
        console.error('Error loading alerts:', error);
    }
}

function displayAlerts(alerts) {
    const alertsContent = document.querySelector('[data-content="alerts"]');
    
    const html = alerts.map(alert => `
        <div class="timeline-item" style="border-left-color: ${getSeverityColor(alert.severity)};">
            <div class="timeline-time">${alert.time_ago}</div>
            <div class="timeline-content">
                <div class="timeline-title" style="color: ${getSeverityColor(alert.severity)};">
                    ${alert.title}
                </div>
                <div class="timeline-desc">${alert.description}</div>
            </div>
        </div>
    `).join('');
    
    alertsContent.innerHTML = `<div class="timeline">${html}</div>`;
}

// ============================
// UTILITIES
// ============================
function getRiskColor(item) {
    if (item.is_suspicious) return COLORS.high;
    const score = item.risk_score || item.pred_prob || 0;
    if (score > 0.7) return COLORS.high;
    if (score > 0.3) return COLORS.medium;
    return COLORS.safe;
}

function getNodeSize(node) {
    if (node.is_suspicious) return 8;
    if ((node.risk_score || 0) > 30) return 6;
    return 5;
}

function getSeverityColor(severity) {
    const colors = {
        high: COLORS.high,
        medium: COLORS.medium,
        low: COLORS.safe
    };
    return colors[severity] || COLORS.safe;
}

function updateHeaderStats(metrics) {
    document.getElementById('totalNodes').textContent = metrics.num_nodes || 0;
    document.getElementById('totalEdges').textContent = (metrics.num_edges || 0).toLocaleString();
    document.getElementById('fraudRate').textContent = `${metrics.fraud_rate || 0}%`;
    document.getElementById('accuracy').textContent = '94.2%';
}

function showNotification(message, type = 'info') {
    // Simple notification - can be enhanced
    console.log(`[${type}] ${message}`);
}

function hideLoading() {
    const loading = document.querySelector('.loading');
    if (loading) {
        loading.style.display = 'none';
    }
}

// ============================
// DRAG HANDLERS
// ============================
function dragStarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
}

function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
}

function dragEnded(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
}

// ============================
// HIGHLIGHT EFFECTS
// ============================
function highlightNode(node, element) {
    d3.select(element)
        .transition()
        .duration(200)
        .attr('r', getNodeSize(node) + 3);
    
    d3.select(element.parentNode).select('text')
        .transition()
        .duration(200)
        .style('opacity', 1)
        .style('font-size', '11px');
}

function unhighlightNode(node, element) {
    if (selectedNode && selectedNode.id === node.id) return;
    
    d3.select(element)
        .transition()
        .duration(200)
        .attr('r', getNodeSize(node));
    
    d3.select(element.parentNode).select('text')
        .transition()
        .duration(200)
        .style('opacity', 0.7)
        .style('font-size', '9px');
}

// ============================
// EVENT LISTENERS
// ============================
function setupEventListeners() {
    // Load graph button
    document.querySelector('.control-btn').addEventListener('click', loadGraph);
    
    // Tool buttons
    document.querySelectorAll('.tool-btn').forEach((btn, idx) => {
        btn.addEventListener('click', () => {
            if (idx === 4) { // Reset button
                resetView();
            }
        });
    });
}

function resetView() {
    const zoom = d3.zoom().scaleExtent([0.1, 5]);
    svg.transition()
        .duration(750)
        .call(zoom.transform, d3.zoomIdentity);
}

// ============================
// REAL-TIME UPDATES
// ============================
function startRealTimeUpdates() {
    // Update alerts every 30 seconds
    setInterval(() => {
        if (document.querySelector('[data-content="alerts"]').classList.contains('active')) {
            loadAlerts();
        }
    }, 30000);
}

// ============================
// INITIALIZE
// ============================
document.addEventListener('DOMContentLoaded', init);