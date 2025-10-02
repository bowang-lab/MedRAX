'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import {
    Upload, Bot, Loader2, FileText, Activity, Eye, X, ChevronRight
} from 'lucide-react';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

interface Message {
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: Date;
    isToolLog?: boolean;
}

interface PatientInfo {
    name: string;
    age: string;
    gender: string;
    notes: string;
}

export default function MedRAXPlatform() {
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [messages, setMessages] = useState<Message[]>([]);
    const [inputMessage, setInputMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [uploadedImages, setUploadedImages] = useState<string[]>([]);
    const [currentImageIndex, setCurrentImageIndex] = useState(0);
    const [dragActive, setDragActive] = useState(false);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [analysisResults, setAnalysisResults] = useState<any[]>([]);
    const [showImageModal, setShowImageModal] = useState(false);
    const [patientInfo, setPatientInfo] = useState<PatientInfo>({
        name: '',
        age: '',
        gender: '',
        notes: ''
    });
    const [showPatientForm, setShowPatientForm] = useState(false);

    const fileInputRef = useRef<HTMLInputElement>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const currentImage = uploadedImages[currentImageIndex] || null;

    // Auto-scroll messages
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // Create session on mount
    useEffect(() => {
        createSession();
    }, []);

    const createSession = async (): Promise<string | null> => {
        try {
            const response = await axios.post(`${API_BASE}/api/sessions`);
            setSessionId(response.data.session_id);
            console.log('✅ Session created:', response.data.session_id);
            return response.data.session_id;
        } catch (error: any) {
            console.error('Failed to create session:', error);
            alert('Failed to create session. Is the backend running?');
            return null;
        }
    };

    const startNewPatient = async () => {
        // Clear all state
        setMessages([]);
        setUploadedImages([]);
        setCurrentImageIndex(0);
        setAnalysisResults([]);
        setShowPatientForm(false);

        // Create new session
        await createSession();

        // Add welcome message
        setMessages([{
            role: 'system',
            content: `🏥 New patient case started${patientInfo.name ? ` for ${patientInfo.name}` : ''}. Upload images to begin analysis.`,
            timestamp: new Date()
        }]);
    };

    const uploadFile = async (file: File) => {
        // Ensure session exists before uploading
        let currentSessionId = sessionId;
        if (!currentSessionId) {
            currentSessionId = await createSession();
            if (!currentSessionId) {
                console.error('Failed to create session');
                return;
            }
        }

        setIsLoading(true);

        // Add uploading message
        setMessages(prev => [...prev, {
            role: 'system',
            content: `📤 Uploading ${file.name}...`,
            timestamp: new Date()
        }]);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post(
                `${API_BASE}/api/upload/${currentSessionId}`,
                formData,
                { headers: { 'Content-Type': 'multipart/form-data' } }
            );

            console.log('✅ Upload successful:', response.data);

            setUploadedImages(prev => {
                const newImages = [...prev, response.data.display_path];
                setCurrentImageIndex(newImages.length - 1);
                return newImages;
            });

            // Update to success message
            setMessages(prev => {
                const newMessages = [...prev];
                newMessages[newMessages.length - 1] = {
                    role: 'system',
                    content: `✅ ${file.name} uploaded successfully! Ready for analysis.`,
                    timestamp: new Date()
                };
                return newMessages;
            });

        } catch (error) {
            console.error('❌ Upload failed:', error);
            setMessages(prev => [...prev, {
                role: 'system',
                content: `❌ Failed to upload ${file.name}`,
                timestamp: new Date()
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const runCompleteAnalysis = async () => {
        if (!sessionId || !currentImage || isAnalyzing) return;

        setIsAnalyzing(true);
        setMessages(prev => [...prev, {
            role: 'system',
            content: '🤖 Starting comprehensive AI analysis...',
            timestamp: new Date()
        }]);

        try {
            // Use the AGENT to decide what to do
            const response = await axios.post(`${API_BASE}/api/chat/${sessionId}`, {
                message: "Please perform a comprehensive analysis of this chest X-ray. Classify pathologies, segment anatomical structures, and provide your findings.",
                image_path: currentImage
            });

            console.log('Agent response:', response.data);

            // Add agent response
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: response.data.response,
                timestamp: new Date()
            }]);

            // Collect any tool call results
            if (response.data.tool_calls && response.data.tool_calls.length > 0) {
                response.data.tool_calls.forEach((toolCall: any) => {
                    setMessages(prev => [...prev, {
                        role: 'system',
                        content: `🔧 Tool: ${toolCall.name}\n${JSON.stringify(toolCall.result, null, 2)}`,
                        timestamp: new Date(),
                        isToolLog: true
                    }]);
                });
            }

            // Get comprehensive results
            const resultsResponse = await axios.get(`${API_BASE}/api/analysis/${sessionId}`);
            setAnalysisResults(Object.entries(resultsResponse.data.results || {}));

        } catch (error: any) {
            console.error('Analysis failed:', error);
            setMessages(prev => [...prev, {
                role: 'system',
                content: `❌ Analysis failed: ${error.response?.data?.detail || error.message}`,
                timestamp: new Date()
            }]);
        } finally {
            setIsAnalyzing(false);
        }
    };

    const sendMessage = async () => {
        if (!inputMessage.trim() || isLoading || !sessionId) return;

        const userMessage = inputMessage;
        setInputMessage('');

        setMessages(prev => [...prev, {
            role: 'user',
            content: userMessage,
            timestamp: new Date()
        }]);

        setIsLoading(true);

        try {
            const response = await axios.post(`${API_BASE}/api/chat/${sessionId}`, {
                message: userMessage,
                image_path: currentImage
            });

            setMessages(prev => [...prev, {
                role: 'assistant',
                content: response.data.response,
                timestamp: new Date()
            }]);

        } catch (error) {
            console.error('Chat failed:', error);
            setMessages(prev => [...prev, {
                role: 'system',
                content: `❌ Failed to get response`,
                timestamp: new Date()
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setDragActive(false);

        const files = Array.from(e.dataTransfer.files);
        files.forEach(file => {
            if (file.type.startsWith('image/') || file.name.endsWith('.dcm')) {
                uploadFile(file);
            }
        });
    }, [sessionId]);

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = Array.from(e.target.files || []);
        files.forEach(file => uploadFile(file));
    };

    return (
        <div className="h-screen bg-zinc-900 text-white flex flex-col">
            {/* Top Bar */}
            <div className="h-16 border-b border-zinc-800 px-6 flex items-center justify-between bg-zinc-950">
                <div className="flex items-center gap-4">
                    <div className="bg-blue-600 rounded-lg p-2">
                        <Activity className="h-5 w-5" />
                    </div>
                    <div>
                        <h1 className="font-semibold">MedRAX Analysis Platform</h1>
                        <p className="text-xs text-zinc-400">
                            {patientInfo.name ? `Patient: ${patientInfo.name}${patientInfo.age ? `, ${patientInfo.age}yo` : ''}` : 'AI-Powered Chest X-Ray Analysis'}
                        </p>
                    </div>
                </div>
                <div className="flex items-center gap-3">
                    <button
                        onClick={() => setShowPatientForm(!showPatientForm)}
                        className="px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-xs font-medium border border-zinc-700"
                    >
                        {patientInfo.name ? '👤 Patient Info' : '➕ Add Patient Info'}
                    </button>
                    <button
                        onClick={startNewPatient}
                        className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 rounded-lg text-xs font-medium"
                    >
                        🏥 New Patient
                    </button>
                    {sessionId && (
                        <div className="text-xs text-zinc-500">
                            Session: {sessionId.slice(0, 8)}
                        </div>
                    )}
                    <div className={`px-3 py-1 rounded-full text-xs font-medium ${isAnalyzing ? 'bg-yellow-900 text-yellow-300' :
                        'bg-green-900 text-green-300'
                        }`}>
                        {isAnalyzing ? '⏳ Analyzing...' : '✅ Ready'}
                    </div>
                </div>
            </div>

            {/* Patient Info Form (Collapsible) */}
            {showPatientForm && (
                <div className="bg-zinc-900 border-b border-zinc-800 p-4">
                    <div className="max-w-2xl">
                        <h3 className="text-sm font-semibold mb-3">Patient Information</h3>
                        <div className="grid grid-cols-3 gap-3">
                            <input
                                type="text"
                                placeholder="Patient Name"
                                value={patientInfo.name}
                                onChange={(e) => setPatientInfo({ ...patientInfo, name: e.target.value })}
                                className="bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                            />
                            <input
                                type="text"
                                placeholder="Age"
                                value={patientInfo.age}
                                onChange={(e) => setPatientInfo({ ...patientInfo, age: e.target.value })}
                                className="bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                            />
                            <select
                                value={patientInfo.gender}
                                onChange={(e) => setPatientInfo({ ...patientInfo, gender: e.target.value })}
                                className="bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                            >
                                <option value="">Gender</option>
                                <option value="Male">Male</option>
                                <option value="Female">Female</option>
                                <option value="Other">Other</option>
                            </select>
                        </div>
                        <textarea
                            placeholder="Clinical notes (optional)"
                            value={patientInfo.notes}
                            onChange={(e) => setPatientInfo({ ...patientInfo, notes: e.target.value })}
                            className="w-full mt-3 bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                            rows={2}
                        />
                    </div>
                </div>
            )}

            <div className="flex-1 flex overflow-hidden">
                {/* Left Sidebar - Uploaded Images */}
                <div className="w-64 border-r border-zinc-800 bg-zinc-950 flex flex-col">
                    <div className="p-4 border-b border-zinc-800">
                        <h3 className="font-medium text-sm mb-3">Uploaded Images</h3>
                        <button
                            onClick={() => fileInputRef.current?.click()}
                            className="w-full bg-blue-600 hover:bg-blue-700 text-white px-3 py-2 rounded-lg text-sm font-medium flex items-center justify-center gap-2"
                        >
                            <Upload className="h-4 w-4" />
                            Upload Image
                        </button>
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".png,.jpg,.jpeg,.dcm"
                            multiple
                            onChange={handleFileSelect}
                            className="hidden"
                        />
                    </div>

                    <div className="flex-1 overflow-y-auto p-4 space-y-2">
                        {uploadedImages.length === 0 ? (
                            <div className="text-center text-zinc-500 text-sm mt-8">
                                <Upload className="h-8 w-8 mx-auto mb-2 opacity-50" />
                                <p>No images uploaded</p>
                            </div>
                        ) : (
                            uploadedImages.map((img, idx) => (
                                <div
                                    key={idx}
                                    onClick={() => setCurrentImageIndex(idx)}
                                    className={`cursor-pointer rounded-lg overflow-hidden border-2 transition-all ${idx === currentImageIndex
                                        ? 'border-blue-500 ring-2 ring-blue-500/20'
                                        : 'border-zinc-700 hover:border-zinc-600'
                                        }`}
                                >
                                    <img
                                        src={`${API_BASE}/${img}`}
                                        alt={`Upload ${idx + 1}`}
                                        className="w-full h-32 object-cover"
                                    />
                                    <div className="p-2 bg-zinc-900 text-xs">
                                        Image {idx + 1}
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>

                {/* Main Content Area */}
                <div className="flex-1 flex flex-col">
                    {/* Analysis Results Area */}
                    <div className="flex-1 overflow-y-auto p-6">
                        {!currentImage ? (
                            // Welcome / Upload Area
                            <div
                                className={`h-full border-2 border-dashed rounded-xl flex flex-col items-center justify-center transition-colors ${dragActive
                                    ? 'border-blue-500 bg-blue-500/10'
                                    : 'border-zinc-700 bg-zinc-900/50'
                                    }`}
                                onDrop={handleDrop}
                                onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
                                onDragLeave={() => setDragActive(false)}
                                onClick={() => fileInputRef.current?.click()}
                            >
                                <Upload className="h-16 w-16 text-zinc-600 mb-4" />
                                <p className="text-xl font-medium text-zinc-300 mb-2">
                                    Drop your chest X-ray here
                                </p>
                                <p className="text-sm text-zinc-500">
                                    or click to browse (PNG, JPG, DICOM)
                                </p>
                            </div>
                        ) : analysisResults.length === 0 && !isAnalyzing ? (
                            // Image uploaded, ready for analysis
                            <div className="h-full flex items-center justify-center">
                                <div className="text-center max-w-2xl">
                                    <div className="mb-6 relative inline-block">
                                        <img
                                            src={`${API_BASE}/${currentImage}`}
                                            alt="Uploaded X-ray"
                                            className="max-h-64 rounded-lg border border-zinc-700 cursor-pointer hover:opacity-80"
                                            onClick={() => setShowImageModal(true)}
                                        />
                                        <button
                                            onClick={() => setShowImageModal(true)}
                                            className="absolute top-2 right-2 bg-black/70 hover:bg-black text-white p-2 rounded-lg"
                                        >
                                            <Eye className="h-4 w-4" />
                                        </button>
                                    </div>
                                    <h2 className="text-2xl font-semibold mb-4">
                                        Image Ready for Analysis
                                    </h2>
                                    <p className="text-zinc-400 mb-6">
                                        Click the button below to let the AI agent analyze this chest X-ray.
                                        The agent will automatically determine which tools to use.
                                    </p>
                                    <button
                                        onClick={runCompleteAnalysis}
                                        disabled={isAnalyzing}
                                        className="bg-blue-600 hover:bg-blue-700 disabled:bg-zinc-700 disabled:cursor-not-allowed text-white px-8 py-4 rounded-xl text-lg font-medium flex items-center gap-3 mx-auto shadow-lg"
                                    >
                                        {isAnalyzing ? (
                                            <>
                                                <Loader2 className="h-6 w-6 animate-spin" />
                                                Analyzing...
                                            </>
                                        ) : (
                                            <>
                                                <Bot className="h-6 w-6" />
                                                Run AI Analysis
                                            </>
                                        )}
                                    </button>
                                </div>
                            </div>
                        ) : (
                            // Show analysis results
                            <div className="space-y-6">
                                {/* Original Image - Keep Visible */}
                                {currentImage && (
                                    <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-6">
                                        <div className="flex items-center justify-between mb-4">
                                            <h3 className="text-lg font-semibold flex items-center gap-2">
                                                <ChevronRight className="h-5 w-5 text-blue-500" />
                                                Original Image
                                            </h3>
                                            <button
                                                onClick={() => setShowImageModal(true)}
                                                className="text-sm text-zinc-400 hover:text-white flex items-center gap-2"
                                            >
                                                <Eye className="h-4 w-4" />
                                                View Full Size
                                            </button>
                                        </div>
                                        <img
                                            src={`${API_BASE}/${currentImage}`}
                                            alt="Original X-ray"
                                            className="max-w-md rounded-lg border border-zinc-700 cursor-pointer hover:opacity-80"
                                            onClick={() => setShowImageModal(true)}
                                        />
                                    </div>
                                )}

                                <h2 className="text-2xl font-semibold">Analysis Results</h2>

                                {analysisResults.map(([toolName, result]: [string, any], idx) => {
                                    const isClassification = toolName.includes('classification') || toolName.includes('classifier');
                                    const isSegmentation = toolName.includes('segmentation');
                                    const isExpert = toolName.includes('expert');

                                    return (
                                        <div key={idx} className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-6">
                                            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                                <ChevronRight className="h-5 w-5 text-blue-500" />
                                                {toolName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                            </h3>

                                            {/* Classification Results - Pretty Format */}
                                            {isClassification && result.result && typeof result.result === 'object' && (
                                                <div className="space-y-2">
                                                    {Object.entries(result.result)
                                                        .filter(([_, prob]) => typeof prob === 'number')
                                                        .sort(([, a], [, b]) => (b as number) - (a as number))
                                                        .slice(0, 8)
                                                        .map(([pathology, prob]) => {
                                                            const probability = (prob as number) * 100;
                                                            const color = probability > 60 ? 'bg-red-500' : probability > 40 ? 'bg-yellow-500' : 'bg-blue-500';
                                                            return (
                                                                <div key={pathology} className="flex items-center gap-3">
                                                                    <span className="w-48 text-sm text-zinc-300">{pathology.replace(/_/g, ' ')}</span>
                                                                    <div className="flex-1 bg-zinc-800 rounded-full h-6 overflow-hidden">
                                                                        <div
                                                                            className={`${color} h-6 rounded-full transition-all duration-500 flex items-center justify-end pr-2`}
                                                                            style={{ width: `${probability}%` }}
                                                                        >
                                                                            {probability > 15 && (
                                                                                <span className="text-xs font-medium text-white">{probability.toFixed(1)}%</span>
                                                                            )}
                                                                        </div>
                                                                    </div>
                                                                    {probability <= 15 && (
                                                                        <span className="text-xs font-mono text-zinc-400 w-16">{probability.toFixed(1)}%</span>
                                                                    )}
                                                                </div>
                                                            );
                                                        })}
                                                    <button
                                                        onClick={() => {
                                                            const el = document.getElementById(`raw-${idx}`);
                                                            if (el) el.classList.toggle('hidden');
                                                        }}
                                                        className="mt-4 text-xs text-zinc-500 hover:text-zinc-300"
                                                    >
                                                        Show all pathologies / raw data
                                                    </button>
                                                    <pre id={`raw-${idx}`} className="hidden bg-zinc-950 p-4 rounded-lg text-xs overflow-x-auto mt-2">
                                                        {JSON.stringify(result, null, 2)}
                                                    </pre>
                                                </div>
                                            )}

                                            {/* Segmentation Results - Pretty Format */}
                                            {isSegmentation && result.result && (
                                                <div className="space-y-4">
                                                    {/* Side-by-side comparison */}
                                                    {result.metadata?.segmentation_image_path && currentImage && (
                                                        <div className="grid grid-cols-2 gap-4 mb-4">
                                                            <div>
                                                                <h4 className="text-xs font-semibold mb-2 text-zinc-400">Original</h4>
                                                                <img
                                                                    src={`${API_BASE}/${currentImage}`}
                                                                    alt="Original X-ray"
                                                                    className="w-full rounded-lg border border-zinc-700"
                                                                />
                                                            </div>
                                                            <div>
                                                                <h4 className="text-xs font-semibold mb-2 text-zinc-400">Segmented</h4>
                                                                <img
                                                                    src={`${API_BASE}/${result.metadata.segmentation_image_path}`}
                                                                    alt="Segmented X-ray"
                                                                    className="w-full rounded-lg border border-zinc-700"
                                                                />
                                                            </div>
                                                        </div>
                                                    )}

                                                    {/* Organ Metrics */}
                                                    {result.result.metrics && (
                                                        <div>
                                                            <h4 className="text-sm font-semibold mb-3 text-zinc-400">Detected Organs:</h4>
                                                            <div className="grid grid-cols-2 gap-3">
                                                                {Object.entries(result.result.metrics).map(([organ, metrics]: [string, any]) => (
                                                                    <div key={organ} className="bg-zinc-800/50 rounded-lg p-3 border border-zinc-700">
                                                                        <div className="font-medium text-sm mb-1">{organ}</div>
                                                                        <div className="text-xs text-zinc-400 space-y-1">
                                                                            <div>Area: <span className="text-white font-mono">{metrics.area_cm2?.toFixed(2)} cm²</span></div>
                                                                            <div>Confidence: <span className="text-white font-mono">{(metrics.confidence_score * 100).toFixed(1)}%</span></div>
                                                                        </div>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                            <button
                                                                onClick={() => {
                                                                    const el = document.getElementById(`raw-${idx}`);
                                                                    if (el) el.classList.toggle('hidden');
                                                                }}
                                                                className="mt-4 text-xs text-zinc-500 hover:text-zinc-300"
                                                            >
                                                                Show detailed metrics / raw data
                                                            </button>
                                                            <pre id={`raw-${idx}`} className="hidden bg-zinc-950 p-4 rounded-lg text-xs overflow-x-auto mt-2">
                                                                {JSON.stringify(result, null, 2)}
                                                            </pre>
                                                        </div>
                                                    )}
                                                </div>
                                            )}

                                            {/* Expert/VQA Results - Clean Format */}
                                            {isExpert && result.result && (
                                                <div>
                                                    <div className="bg-zinc-800/30 rounded-lg p-4 border-l-4 border-blue-500">
                                                        <p className="text-sm leading-relaxed">
                                                            {typeof result.result === 'string'
                                                                ? result.result
                                                                : result.result.response || JSON.stringify(result.result)}
                                                        </p>
                                                    </div>
                                                </div>
                                            )}

                                            {/* Fallback for other tools */}
                                            {!isClassification && !isSegmentation && !isExpert && (
                                                <pre className="bg-zinc-950 p-4 rounded-lg text-xs overflow-x-auto">
                                                    {JSON.stringify(result, null, 2)}
                                                </pre>
                                            )}
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </div>

                    {/* Bottom Chat/Log Area */}
                    <div className="border-t border-zinc-800 bg-zinc-950">
                        <div className="p-4">
                            <div className="flex items-center gap-2 mb-3">
                                <Bot className="h-4 w-4 text-blue-500" />
                                <h3 className="text-sm font-medium">AI Agent Logs & Chat</h3>
                            </div>

                            {/* Messages */}
                            <div className="bg-zinc-900 rounded-lg p-4 h-48 overflow-y-auto mb-3 space-y-2">
                                {messages.length === 0 ? (
                                    <p className="text-sm text-zinc-500 text-center mt-8">
                                        Agent logs and conversation will appear here
                                    </p>
                                ) : (
                                    messages.map((msg, idx) => (
                                        <div
                                            key={idx}
                                            className={`text-sm p-2 rounded ${msg.role === 'user'
                                                ? 'bg-blue-900/30 text-blue-100'
                                                : msg.role === 'system'
                                                    ? msg.isToolLog
                                                        ? 'bg-purple-900/20 text-purple-200 font-mono text-xs'
                                                        : 'bg-zinc-800 text-zinc-300'
                                                    : 'bg-green-900/20 text-green-100'
                                                }`}
                                        >
                                            <div className="flex items-start gap-2">
                                                <span className="text-xs opacity-60 min-w-[60px]">
                                                    {msg.timestamp.toLocaleTimeString()}
                                                </span>
                                                <span className="flex-1 whitespace-pre-wrap">{msg.content}</span>
                                            </div>
                                        </div>
                                    ))
                                )}
                                <div ref={messagesEndRef} />
                            </div>

                            {/* Chat Input */}
                            <div className="flex gap-2">
                                <input
                                    type="text"
                                    value={inputMessage}
                                    onChange={(e) => setInputMessage(e.target.value)}
                                    onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                                    placeholder="Ask the AI agent anything..."
                                    disabled={!sessionId || isLoading}
                                    className="flex-1 bg-zinc-800 border border-zinc-700 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                                />
                                <button
                                    onClick={sendMessage}
                                    disabled={!inputMessage.trim() || isLoading}
                                    className="bg-blue-600 hover:bg-blue-700 disabled:bg-zinc-700 disabled:cursor-not-allowed text-white px-6 py-2 rounded-lg text-sm font-medium"
                                >
                                    Send
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Full Image Modal */}
            {showImageModal && currentImage && (
                <div
                    className="fixed inset-0 bg-black/90 flex items-center justify-center z-50 p-8"
                    onClick={() => setShowImageModal(false)}
                >
                    <button
                        onClick={() => setShowImageModal(false)}
                        className="absolute top-4 right-4 bg-white/10 hover:bg-white/20 text-white p-2 rounded-lg"
                    >
                        <X className="h-6 w-6" />
                    </button>
                    <img
                        src={`${API_BASE}/${currentImage}`}
                        alt="Full size X-ray"
                        className="max-h-full max-w-full object-contain"
                        onClick={(e) => e.stopPropagation()}
                    />
                </div>
            )}
        </div>
    );
}

