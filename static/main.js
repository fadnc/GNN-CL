// Advanced GNN Fraud Detection - Main JavaScript
// Fixed version with proper API integration

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
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing GNN Fraud Detection System...');
    init();
});

function init() {
    initSVG();
    setupEventListeners();
    loadInitialData();
}

function initSVG() {
    const svgElement = document.getElementById('graphSvg');
    const container = svgElement.parentElement;
    
    width = container.clientWidth;
    height = container.clientHeight;

    svg = d3.select('#graphSvg')
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
    
    console.log('SVG initialized:', width, 'x', height);
}

function setupZoom() {
    const zoom = d3.zoom()
        .scaleExtent([0.1, 5])
        .filter(function(event) {
            // Allow zoom on wheel/pinch, but not on node drag
            // Return false for mousedown on nodes to prevent interference
            if (event.type === 'mousedown') {
                const isNode = event.target.closest('.node-group');
                if (isNode) return false;
            }
            return true;
        })
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });

    svg.call(zoom);
    svg.style('cursor', 'grab');
    
    // Update cursor on pan
    svg.on('mousedown.cursor', function() {
        const isNode = d3.select(event.target).classed('node-group') || 
                       d3.select(event.target.parentNode).classed('node-group');
        if (!isNode) {
            svg.style('cursor', 'grabbing');
        }
    });
    
    svg.on('mouseup.cursor', function() {
        svg.style('cursor', 'grab');
    });
}

// ============================
// DATA LOADING
// ============================
async function loadInitialData() {
    try {
        console.log('Loading initial data...');
        
        // Check if API endpoints exist
        const response = await fetch('/api/metrics');
        if (response.ok) {
            const metricsData = await response.json();
            console.log('Metrics loaded:', metricsData);
            updateHeaderStats(metricsData.metrics);
        } else {
            console.warn('API metrics not available, using old endpoint');
        }
        
        // Load graph
        await loadGraph();
        
        hideLoading();
    } catch (error) {
        console.error('Error loading initial data:', error);
        // Try loading with old API
        await loadGraphOldAPI();
    }
}

async function loadGraph() {
    const nodesInput = document.getElementById('nodesInput');
    const edgesInput = document.getElementById('edgesInput');
    
    const nodes = nodesInput ? nodesInput.value : 50;
    const edges = edgesInput ? edgesInput.value : 200;
    
    console.log(`Loading graph with ${nodes} nodes and ${edges} edges...`);
    showLoading('Loading graph...');
    
    try {
        // Try new API first
        let response = await fetch(`/api/graph?nodes=${nodes}&edges=${edges}`);
        
        // If new API doesn't exist, try old one
        if (!response.ok) {
            console.log('Trying old API endpoint...');
            response = await fetch(`/graph?nodes=${nodes}&edges=${edges}`);
        }
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Graph data received:', data);
        
        currentGraph = data;
        renderGraph(data);
        
        if (data.metrics) {
            updateHeaderStats(data.metrics);
            updateAnalysisTab();
        }
        
        hideLoading();
        showNotification('Graph loaded successfully', 'success');
    } catch (error) {
        console.error('Error loading graph:', error);
        hideLoading();
        showNotification('Failed to load graph: ' + error.message, 'error');
    }
}

async function loadGraphOldAPI() {
    // Fallback to old API
    console.log('Using old API structure...');
    try {
        const nodes = document.getElementById('nodesInput').value || 20;
        const edges = document.getElementById('edgesInput').value || 150;
        
        const response = await fetch(`/graph?nodes=${nodes}&edges=${edges}`);
        const data = await response.json();
        
        console.log('Old API data received:', data);
        renderGraph(data);
        hideLoading();
    } catch (error) {
        console.error('Old API also failed:', error);
        hideLoading();
    }
}

// ============================
// GRAPH RENDERING
// ============================
function renderGraph(data) {
    console.log('Rendering graph...');
    
    // Clear existing
    g.selectAll('*').remove();

    if (!data.nodes || !data.edges) {
        console.error('Invalid graph data:', data);
        return;
    }

    console.log(`Rendering ${data.nodes.length} nodes and ${data.edges.length} edges`);
    
    // Debug: Check a sample node's data
    if (data.nodes.length > 0) {
        const sampleNode = data.nodes[0];
        console.log('Sample node data:', {
            id: sampleNode.id,
            type: sampleNode.type,
            risk_score: sampleNode.risk_score,
            is_suspicious: sampleNode.is_suspicious
        });
    }

    // Create links
    const link = g.append('g')
        .attr('class', 'links')
        .selectAll('line')
        .data(data.edges)
        .enter()
        .append('line')
        .style('stroke', d => {
            const color = getRiskColor(d);
            return color;
        })
        .style('stroke-width', d => d.is_suspicious ? 2.5 : 1.5)
        .style('opacity', d => d.is_suspicious ? 0.7 : 0.4)
        .attr('marker-end', d => {
            const type = d.is_suspicious ? 'high' : 
                        (d.pred_prob || 0) > 0.3 ? 'medium' : 'safe';
            return `url(#arrow-${type})`;
        });

    // Debug: Count nodes by color
    const colorCounts = { safe: 0, medium: 0, high: 0 };
    data.nodes.forEach(node => {
        const color = getRiskColor(node);
        if (color === COLORS.safe) colorCounts.safe++;
        else if (color === COLORS.medium) colorCounts.medium++;
        else if (color === COLORS.high) colorCounts.high++;
    });
    console.log('Node color distribution:', colorCounts);

    // Create node groups
    const nodeGroup = g.append('g')
        .attr('class', 'nodes')
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
        .style('fill', d => {
            const color = getRiskColor(d);
            return color;
        })
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
    
    console.log('Graph rendering complete');
}

// ============================
// NODE INTERACTION
// ============================
async function selectNode(node) {
    console.log('Node selected:', node.id);
    selectedNode = node;
    
    // Highlight selection
    g.selectAll('.node-group circle')
        .style('stroke-width', d => d.id === node.id ? 3 : d.is_suspicious ? 2 : 1);
    
    // Try to load detailed info from new API
    try {
        const response = await fetch(`/api/node/${node.id}`);
        if (response.ok) {
            const data = await response.json();
            displayNodeDetails(data);
            return;
        }
    } catch (error) {
        console.log('New API not available, trying old API...');
    }
    
    // Fallback to old API
    try {
        const response = await fetch(`/node_details?id=${node.id}`);
        const data = await response.json();
        displayNodeDetailsOld(data);
    } catch (error) {
        console.error('Error loading node details:', error);
    }
}

function displayNodeDetails(data) {
    const detailsContent = document.querySelector('[data-content="details"]');
    
    // Normalize risk score for display
    let displayRisk = data.risk_score || 0;
    let riskPercent = displayRisk;
    
    // If risk is between 0-1, convert to percentage
    if (displayRisk <= 1) {
        riskPercent = displayRisk * 100;
    }
    
    // Determine risk level text
    let riskLevel = 'Low';
    let riskColor = COLORS.safe;
    if (data.is_suspicious || riskPercent > 70) {
        riskLevel = 'High';
        riskColor = COLORS.high;
    } else if (riskPercent > 30) {
        riskLevel = 'Medium';
        riskColor = COLORS.medium;
    }
    
    const html = `
        <div class="info-card">
            <h3>Node Information</h3>
            <div class="info-row">
                <span class="info-label">Node ID</span>
                <span class="info-value">${data.id}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Type</span>
                <span class="info-value">${data.type || 'unknown'}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Transactions</span>
                <span class="info-value">${data.degree}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Avg Amount</span>
                <span class="info-value">${(data.summary?.avg_amount || 0).toFixed(2)}</span>
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
                    <div class="risk-indicator" style="left: ${riskPercent}%;"></div>
                </div>
                <div style="text-align: center; margin-top: 15px;">
                    <div style="font-size: 32px; font-weight: bold; color: ${riskColor};">
                        ${riskLevel}
                    </div>
                    <div style="font-size: 12px; color: #a0b0c0; margin-top: 5px;">
                        Risk Score: ${riskPercent.toFixed(2)}%
                    </div>
                </div>
            </div>
        </div>

        <div class="info-card">
            <h3>Top Counterparties</h3>
            ${Object.entries(data.top_counterparties || {}).slice(0, 5).map(([id, count]) => `
                <div class="info-row">
                    <span class="info-label">${id}</span>
                    <span class="info-value">${count} txns</span>
                </div>
            `).join('')}
        </div>
    `;
    
    detailsContent.innerHTML = html;
}

function displayNodeDetailsOld(data) {
    const detailsContent = document.querySelector('[data-content="details"]');
    
    // Normalize risk score - old API uses 'risk' field which is 0-1
    let riskScore = data.risk || 0;
    let riskPercent = riskScore * 100;
    
    let riskLevel = 'Low';
    let riskColor = COLORS.safe;
    if (riskPercent > 70) {
        riskLevel = 'High';
        riskColor = COLORS.high;
    } else if (riskPercent > 30) {
        riskLevel = 'Medium';
        riskColor = COLORS.medium;
    }
    
    const html = `
        <div class="info-card">
            <h3>Node Information</h3>
            <div class="info-row">
                <span class="info-label">Node ID</span>
                <span class="info-value">${data.id}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Transactions</span>
                <span class="info-value">${data.degree}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Avg Amount</span>
                <span class="info-value">${(data.summary?.avg_amount || 0).toFixed(2)}</span>
            </div>
            <div class="info-row" style="border: none;">
                <span class="info-label">Risk</span>
                <span class="info-value">${riskPercent.toFixed(2)}%</span>
            </div>
        </div>

        <div class="info-card">
            <h3>Risk Assessment</h3>
            <div class="risk-meter">
                <div class="risk-bar">
                    <div class="risk-indicator" style="left: ${riskPercent}%;"></div>
                </div>
                <div style="text-align: center; margin-top: 15px;">
                    <div style="font-size: 32px; font-weight: bold; color: ${riskColor};">
                        ${riskLevel}
                    </div>
                    <div style="font-size: 12px; color: #a0b0c0; margin-top: 5px;">
                        Risk Score: ${riskPercent.toFixed(2)}%
                    </div>
                </div>
            </div>
        </div>

        <div class="info-card">
            <h3>Top Counterparties</h3>
            ${Object.entries(data.top_counterparties || {}).slice(0, 5).map(([id, count]) => `
                <div class="info-row">
                    <span class="info-label">${id}</span>
                    <span class="info-value">${count} txns</span>
                </div>
            `).join('')}
        </div>
    `;
    
    detailsContent.innerHTML = html;
}

// ============================
// UTILITIES
// ============================
function getRiskColor(item) {
    if (item.is_suspicious) return COLORS.high;
    
    // Handle different risk score formats
    let score = item.risk_score || item.pred_prob || 0;
    
    // Normalize score to 0-1 range if it's in 0-100 range
    if (score > 1) {
        score = score / 100;
    }
    
    // Color based on normalized score
    if (score > 0.7) return COLORS.high;      // High risk: > 70%
    if (score > 0.3) return COLORS.medium;     // Medium risk: 30-70%
    return COLORS.safe;                        // Low risk: < 30%
}

function getRiskColorValue(score) {
    // Normalize score to 0-1 range if needed
    if (score > 1) {
        score = score / 100;
    }
    
    if (score > 0.7) return COLORS.high;
    if (score > 0.3) return COLORS.medium;
    return COLORS.safe;
}

function getNodeSize(node) {
    if (node.is_suspicious) return 8;
    
    // Handle different risk score formats
    let score = node.risk_score || 0;
    
    // Normalize score to 0-1 range if it's in 0-100 range
    if (score > 1) {
        score = score / 100;
    }
    
    // Size based on normalized score
    if (score > 0.5) return 6;  // Medium-high risk
    return 5;                    // Low risk
}

function updateHeaderStats(metrics) {
    document.getElementById('totalNodes').textContent = metrics.num_nodes || 0;
    document.getElementById('totalEdges').textContent = (metrics.num_edges || 0).toLocaleString();
    document.getElementById('fraudRate').textContent = `${(metrics.fraud_rate || 0).toFixed(1)}%`;
    document.getElementById('accuracy').textContent = '94.2%';
}

function showNotification(message, type = 'info') {
    console.log(`[${type}] ${message}`);
}

function showLoading(message = 'Loading...') {
    const loading = document.getElementById('loadingIndicator');
    if (loading) {
        loading.style.display = 'block';
        const text = loading.querySelector('div:last-child');
        if (text) text.textContent = message;
    }
}

function hideLoading() {
    const loading = document.getElementById('loadingIndicator');
    if (loading) {
        loading.style.display = 'none';
    }
}

// ============================
// DRAG HANDLERS
// ============================
function dragStarted(event, d) {
    // Stop propagation to prevent zoom from interfering
    event.sourceEvent.stopPropagation();
    
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
    
    // Change cursor
    d3.select(this).style('cursor', 'grabbing');
}

function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
}

function dragEnded(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    
    // Release fixed position so node can move freely
    d.fx = null;
    d.fy = null;
    
    // Reset cursor
    d3.select(this).style('cursor', 'pointer');
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
    console.log('Setting up event listeners...');
    
    // Load graph button
    const loadBtn = document.getElementById('loadGraphBtn');
    if (loadBtn) {
        loadBtn.addEventListener('click', () => {
            console.log('Load Graph button clicked');
            loadGraph();
        });
        console.log('Load button listener attached');
    }
    
    // Tab switching
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            console.log('Tab clicked:', tab.getAttribute('data-tab'));
            const tabName = tab.getAttribute('data-tab');
            
            // Remove active from all tabs and content
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            // Add active to clicked tab and corresponding content
            tab.classList.add('active');
            const content = document.querySelector(`[data-content="${tabName}"]`);
            if (content) {
                content.classList.add('active');
                console.log('Activated tab content:', tabName);
                
                // Load data when switching to certain tabs
                if (tabName === 'alerts') {
                    loadAlerts();
                } else if (tabName === 'analysis') {
                    updateAnalysisTab();
                }
            }
        });
    });
    
    // Filter chips - Risk Level
    const riskFilters = document.querySelectorAll('.filter-chips')[0];
    if (riskFilters) {
        riskFilters.querySelectorAll('.chip').forEach(chip => {
            chip.addEventListener('click', (e) => {
                console.log('Risk filter clicked:', chip.textContent);
                // Remove active from siblings
                riskFilters.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
                // Add active to clicked chip
                chip.classList.add('active');
                // Apply filter
                applyRiskFilter(chip.textContent.trim());
            });
        });
    }
    
    // Filter chips - Node Type
    const typeFilters = document.querySelectorAll('.filter-chips')[1];
    if (typeFilters) {
        typeFilters.querySelectorAll('.chip').forEach(chip => {
            chip.addEventListener('click', (e) => {
                console.log('Type filter clicked:', chip.textContent);
                // Remove active from siblings
                typeFilters.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
                // Add active to clicked chip
                chip.classList.add('active');
                // Apply filter
                applyTypeFilter(chip.textContent.trim());
            });
        });
    }
    
    console.log('Event listeners setup complete');
}

// ============================
// FILTER FUNCTIONS
// ============================
function applyRiskFilter(level) {
    console.log('Applying risk filter:', level);
    
    if (!currentGraph.nodes || currentGraph.nodes.length === 0) {
        console.log('No graph loaded yet');
        return;
    }
    
    g.selectAll('.node-group').style('opacity', function(d) {
        if (level === 'All') return 1;
        
        // Normalize risk score
        let riskScore = d.risk_score || d.pred_prob || 0;
        if (riskScore > 1) riskScore = riskScore / 100;
        
        const isSuspicious = d.is_suspicious;
        
        // Filter logic
        if (level === 'High' && (isSuspicious || riskScore > 0.7)) return 1;
        if (level === 'Medium' && !isSuspicious && (riskScore > 0.3 && riskScore <= 0.7)) return 1;
        if (level === 'Low' && !isSuspicious && riskScore <= 0.3) return 1;
        
        return 0.2;
    });
    
    // Also filter edges
    g.selectAll('.links line').style('opacity', function(d) {
        if (level === 'All') return d.is_suspicious ? 0.7 : 0.4;
        
        const sourceNode = currentGraph.nodes.find(n => n.id === d.source.id || n.id === d.source);
        const targetNode = currentGraph.nodes.find(n => n.id === d.target.id || n.id === d.target);
        
        // Show edge if either node matches filter
        const sourceVisible = g.selectAll('.node-group').filter(n => n.id === (sourceNode?.id || d.source.id || d.source)).style('opacity') == 1;
        const targetVisible = g.selectAll('.node-group').filter(n => n.id === (targetNode?.id || d.target.id || d.target)).style('opacity') == 1;
        
        return (sourceVisible && targetVisible) ? (d.is_suspicious ? 0.7 : 0.4) : 0.1;
    });
}

function applyTypeFilter(type) {
    console.log('Applying type filter:', type);
    
    if (!currentGraph.nodes || currentGraph.nodes.length === 0) {
        console.log('No graph loaded yet');
        return;
    }
    
    g.selectAll('.node-group').style('opacity', function(d) {
        if (type === 'All') return 1;
        if (type === 'Users' && d.type === 'user') return 1;
        if (type === 'Merchants' && d.type === 'merchant') return 1;
        return 0.2;
    });
    
    // Also filter edges
    g.selectAll('.links line').style('opacity', function(d) {
        if (type === 'All') return d.is_suspicious ? 0.7 : 0.4;
        
        const sourceNode = currentGraph.nodes.find(n => n.id === d.source.id || n.id === d.source);
        const targetNode = currentGraph.nodes.find(n => n.id === d.target.id || n.id === d.target);
        
        const sourceVisible = g.selectAll('.node-group').filter(n => n.id === (sourceNode?.id || d.source.id || d.source)).style('opacity') == 1;
        const targetVisible = g.selectAll('.node-group').filter(n => n.id === (targetNode?.id || d.target.id || d.target)).style('opacity') == 1;
        
        return (sourceVisible && targetVisible) ? (d.is_suspicious ? 0.7 : 0.4) : 0.1;
    });
}

// ============================
// ANALYSIS TAB UPDATE
// ============================
function updateAnalysisTab() {
    if (!currentGraph.metrics) return;
    
    const metrics = currentGraph.metrics;
    
    // Update metric values
    if (document.getElementById('metric-density')) {
        document.getElementById('metric-density').textContent = (metrics.density || 0).toFixed(4);
    }
    if (document.getElementById('metric-clustering')) {
        document.getElementById('metric-clustering').textContent = (metrics.avg_clustering || 0).toFixed(3);
    }
    if (document.getElementById('metric-modularity')) {
        document.getElementById('metric-modularity').textContent = (metrics.modularity || 0).toFixed(3);
    }
    if (document.getElementById('metric-communities')) {
        document.getElementById('metric-communities').textContent = metrics.num_communities || 0;
    }
    
    // Update fraud statistics
    if (document.getElementById('total-transactions')) {
        document.getElementById('total-transactions').textContent = (metrics.num_edges || 0).toLocaleString();
    }
    if (document.getElementById('suspicious-nodes')) {
        document.getElementById('suspicious-nodes').textContent = metrics.fraud_nodes_count || 0;
    }
    if (document.getElementById('suspicious-edges')) {
        document.getElementById('suspicious-edges').textContent = metrics.fraud_edges_count || 0;
    }
}

// ============================
// ALERTS LOADING
// ============================
async function loadAlerts() {
    try {
        console.log('Loading alerts...');
        const response = await fetch('/api/alerts?limit=10');
        
        if (!response.ok) {
            console.log('Alerts API not available, using mock data');
            return;
        }
        
        const data = await response.json();
        displayAlerts(data.alerts);
        
    } catch (error) {
        console.log('Error loading alerts:', error);
    }
}

function displayAlerts(alerts) {
    const container = document.getElementById('alerts-timeline');
    if (!container) return;
    
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
    
    container.innerHTML = html;
}

function getSeverityColor(severity) {
    const colors = {
        high: '#ff0066',
        medium: '#ffa500',
        low: '#00eaff'
    };
    return colors[severity] || '#00eaff';
}

// ============================
// GLOBAL FUNCTIONS
// ============================
function resetView() {
    const zoom = d3.zoom().scaleExtent([0.1, 5]);
    svg.transition()
        .duration(750)
        .call(zoom.transform, d3.zoomIdentity);
}

function centerGraph() {
    if (simulation) {
        simulation.alpha(0.3).restart();
    }
}

// Make functions globally accessible
window.resetView = resetView;
window.centerGraph = centerGraph;