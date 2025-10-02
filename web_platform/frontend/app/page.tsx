'use client';

import { useState, useRef, useEffect } from 'react';
import { Bot, Loader2, X, Terminal, ChevronDown, ChevronUp } from 'lucide-react';
import axios from 'axios';
import PatientSidebar from '../components/PatientSidebar';
import ClassificationResults from '../components/ClassificationResults';
import SegmentationResults from '../components/SegmentationResults';
import ReportResults from '../components/ReportResults';
import ImageUploadZone from '../components/ImageUploadZone';
import AnalysisProgress from '../components/AnalysisProgress';
import { getAllSessions, saveSession, getSession, SessionData } from '../lib/sessionStorage';

const API_BASE = 'http://localhost:8000';

interface Message {
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: Date;
}

interface PatientInfo {
    name: string;
    age: string;
    gender: string;
    notes: string;
}

export default function MedRAXPlatform() {
    // Core state
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [messages, setMessages] = useState<Message[]>([]);
    const [inputMessage, setInputMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [uploadedImages, setUploadedImages] = useState<string[]>([]);
    const [currentImageIndex, setCurrentImageIndex] = useState(0);
    const [dragActive, setDragActive] = useState(false);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [analysisResults, setAnalysisResults] = useState<any[]>([]);

    // Patient info
    const [patientInfo, setPatientInfo] = useState<PatientInfo>({
        name: '',
        age: '',
        gender: '',
        notes: ''
    });
    const [showPatientForm, setShowPatientForm] = useState(false);

    // Session history
    const [sessionHistory, setSessionHistory] = useState<SessionData[]>([]);

    // Logs
    const [showLogs, setShowLogs] = useState(false);
    const [backendLogs, setBackendLogs] = useState<string[]>([]);

    const fileInputRef = useRef<HTMLInputElement>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const currentImage = uploadedImages[currentImageIndex] || null;

    // Load session history on mount
    useEffect(() => {
        setSessionHistory(getAllSessions());
        createSession();
    }, []);

    // Auto-scroll messages
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // Save session to history when there are changes
    useEffect(() => {
        if (sessionId && (uploadedImages.length > 0 || messages.length > 0)) {
            const sessionData: SessionData = {
                sessionId,
                patientName: patientInfo.name,
                patientAge: patientInfo.age,
                patientGender: patientInfo.gender,
                patientNotes: patientInfo.notes,
                timestamp: new Date().toISOString(),
                imageCount: uploadedImages.length,
                uploadedImages,
                analysisResults,
                messages: messages.map(m => ({ ...m, timestamp: m.timestamp.toISOString() }))
            };
            saveSession(sessionData);
            setSessionHistory(getAllSessions());
        }
    }, [sessionId, uploadedImages, analysisResults, messages.length]);

    const createSession = async (): Promise<string | null> => {
        try {
            const response = await axios.post(`${API_BASE}/api/sessions`);
            setSessionId(response.data.session_id);
            console.log('✅ Session created:', response.data.session_id);
            return response.data.session_id;
        } catch (error: any) {
            console.error('Failed to create session:', error);
            return null;
        }
    };

    const loadSession = (savedSession: SessionData) => {
        setSessionId(savedSession.sessionId);
        setPatientInfo({
            name: savedSession.patientName,
            age: savedSession.patientAge,
            gender: savedSession.patientGender,
            notes: savedSession.patientNotes
        });
        setUploadedImages(savedSession.uploadedImages);
        setAnalysisResults(savedSession.analysisResults);
        setMessages(savedSession.messages.map(m => ({
            ...m,
            timestamp: new Date(m.timestamp)
        })));
        setCurrentImageIndex(0);
    };

    const startNewPatient = async () => {
        setMessages([]);
        setUploadedImages([]);
        setCurrentImageIndex(0);
        setAnalysisResults([]);
        setShowPatientForm(false);
        setPatientInfo({ name: '', age: '', gender: '', notes: '' });

        const newSessionId = await createSession();

        if (newSessionId) {
            setMessages([{
                role: 'system',
                content: `🏥 New patient case started. Upload images to begin analysis.`,
                timestamp: new Date()
            }]);
        }
    };

    const uploadFile = async (file: File) => {
        let currentSessionId = sessionId;
        if (!currentSessionId) {
            currentSessionId = await createSession();
            if (!currentSessionId) return;
        }

        setIsLoading(true);
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

            setUploadedImages(prev => {
                const newImages = [...prev, response.data.display_path];
                setCurrentImageIndex(newImages.length - 1);
                return newImages;
            });

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

        // Clear previous logs
        setBackendLogs([]);
        setBackendLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Starting analysis...`]);

        try {
            // Use fetch for streaming instead of axios
            const response = await fetch(`${API_BASE}/api/chat/${sessionId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: "Please perform a comprehensive analysis of this chest X-ray. Classify pathologies, segment anatomical structures, and provide your findings.",
                    image_path: currentImage
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            const responseText = data.response;

            // Parse response into tool messages and assistant messages
            const lines = responseText.split('\n\n');
            let currentMessage = '';
            let currentRole: 'system' | 'assistant' = 'assistant';

            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed) continue;

                if (trimmed.startsWith('🔧')) {
                    // Flush previous message if any
                    if (currentMessage) {
                        setMessages(prev => [...prev, {
                            role: currentRole,
                            content: currentMessage,
                            timestamp: new Date()
                        }]);
                        currentMessage = '';
                        await new Promise(resolve => setTimeout(resolve, 100));
                    }

                    // Add tool message
                    setMessages(prev => [...prev, {
                        role: 'system',
                        content: trimmed,
                        timestamp: new Date()
                    }]);
                    setBackendLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${trimmed}`]);
                    await new Promise(resolve => setTimeout(resolve, 100));
                } else {
                    // Accumulate assistant message
                    if (currentMessage) {
                        currentMessage += '\n\n' + trimmed;
                    } else {
                        currentMessage = trimmed;
                        currentRole = 'assistant';
                    }
                }
            }

            // Flush final message
            if (currentMessage) {
                setMessages(prev => [...prev, {
                    role: currentRole,
                    content: currentMessage,
                    timestamp: new Date()
                }]);
            }

            // Fetch analysis results
            const resultsResponse = await axios.get(`${API_BASE}/api/analysis/${sessionId}`);
            setAnalysisResults(Object.entries(resultsResponse.data.results || {}));

            setMessages(prev => [...prev, {
                role: 'system',
                content: '✅ Analysis complete! Results displayed above.',
                timestamp: new Date()
            }]);

            setBackendLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ✅ Analysis complete`]);
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
        setIsLoading(true);

        setMessages(prev => [...prev, {
            role: 'user',
            content: userMessage,
            timestamp: new Date()
        }]);

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
                content: '❌ Failed to send message',
                timestamp: new Date()
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    // File handling
    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setDragActive(false);
        const files = Array.from(e.dataTransfer.files);
        files.forEach(file => uploadFile(file));
    };

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = Array.from(e.target.files || []);
        files.forEach(file => uploadFile(file));
    };

    return (
        <div className="h-screen flex bg-zinc-950 text-white">
            {/* Analysis Progress Overlay */}
            <AnalysisProgress isAnalyzing={isAnalyzing} />

            {/* Patient History Sidebar */}
            <PatientSidebar
                sessions={sessionHistory.map((s, idx) => ({
                    sessionId: s.sessionId,
                    patientName: s.patientName,
                    patientAge: s.patientAge,
                    timestamp: new Date(s.timestamp),
                    imageCount: s.imageCount,
                    isActive: s.sessionId === sessionId
                }))}
                onSelectSession={(id) => {
                    const session = getSession(id);
                    if (session) loadSession(session);
                }}
                onNewPatient={startNewPatient}
            />

            {/* Main Content */}
            <div className="flex-1 flex flex-col">
                {/* Header */}
                <div className="bg-zinc-900 border-b border-zinc-800 p-4 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <div className="bg-blue-600 rounded-lg p-2">
                            <Bot className="h-5 w-5" />
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
                        {sessionId && (
                            <div className="text-xs text-zinc-500">
                                Session: {sessionId.slice(0, 8)}
                            </div>
                        )}
                        <div className={`px-3 py-1 rounded-full text-xs font-medium ${isAnalyzing ? 'bg-yellow-900 text-yellow-300' : 'bg-green-900 text-green-300'}`}>
                            {isAnalyzing ? '⏳ Analyzing...' : '✅ Ready'}
                        </div>
                    </div>
                </div>

                {/* Patient Info Form */}
                {showPatientForm && (
                    <div className="bg-zinc-900 border-b border-zinc-800 p-4">
                        <div className="max-w-2xl">
                            <h3 className="text-sm font-semibold mb-3">Patient Information</h3>
                            <div className="grid grid-cols-3 gap-3">
                                <input
                                    type="text"
                                    placeholder="Name"
                                    value={patientInfo.name}
                                    onChange={(e) => setPatientInfo(prev => ({ ...prev, name: e.target.value }))}
                                    className="px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm"
                                />
                                <input
                                    type="text"
                                    placeholder="Age"
                                    value={patientInfo.age}
                                    onChange={(e) => setPatientInfo(prev => ({ ...prev, age: e.target.value }))}
                                    className="px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm"
                                />
                                <select
                                    value={patientInfo.gender}
                                    onChange={(e) => setPatientInfo(prev => ({ ...prev, gender: e.target.value }))}
                                    className="px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm"
                                >
                                    <option value="">Gender</option>
                                    <option value="male">Male</option>
                                    <option value="female">Female</option>
                                    <option value="other">Other</option>
                                </select>
                            </div>
                            <textarea
                                placeholder="Notes..."
                                value={patientInfo.notes}
                                onChange={(e) => setPatientInfo(prev => ({ ...prev, notes: e.target.value }))}
                                className="mt-3 w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm"
                                rows={2}
                            />
                        </div>
                    </div>
                )}

                {/* Main Content Area */}
                <div className="flex-1 flex overflow-hidden">
                    {/* Left: Image Upload */}
                    <div className="w-80 bg-zinc-900/50 border-r border-zinc-800 flex flex-col">
                        <div className="p-4 border-b border-zinc-800">
                            <h2 className="text-sm font-semibold text-zinc-400">Medical Images</h2>
                        </div>

                        <div className="flex-1 overflow-y-auto p-4 space-y-2">
                            {uploadedImages.length === 0 ? (
                                <div className="text-center text-zinc-500 text-sm mt-8">
                                    No images uploaded yet
                                </div>
                            ) : (
                                uploadedImages.map((img, idx) => (
                                    <div
                                        key={idx}
                                        onClick={() => setCurrentImageIndex(idx)}
                                        className={`relative cursor-pointer rounded-lg overflow-hidden border-2 transition-all ${idx === currentImageIndex ? 'border-blue-500' : 'border-zinc-800 hover:border-zinc-700'
                                            }`}
                                    >
                                        <img
                                            src={`${API_BASE}/${img}`}
                                            alt={`Upload ${idx + 1}`}
                                            className="w-full h-32 object-cover"
                                        />
                                    </div>
                                ))
                            )}
                        </div>

                        <div className="p-4 border-t border-zinc-800">
                            <ImageUploadZone
                                dragActive={dragActive}
                                onDragEnter={(e) => { e.preventDefault(); setDragActive(true); }}
                                onDragLeave={(e) => { e.preventDefault(); setDragActive(false); }}
                                onDragOver={(e) => e.preventDefault()}
                                onDrop={handleDrop}
                                onClick={() => fileInputRef.current?.click()}
                            />
                            <input
                                ref={fileInputRef}
                                type="file"
                                multiple
                                accept="image/*,.dcm"
                                onChange={handleFileSelect}
                                className="hidden"
                            />
                        </div>
                    </div>

                    {/* Center: Results */}
                    <div className="flex-1 overflow-y-auto p-6">
                        {uploadedImages.length === 0 ? (
                            <div className="flex items-center justify-center h-full">
                                <div className="text-center text-zinc-600">
                                    <Bot className="h-16 w-16 mx-auto mb-4 opacity-50" />
                                    <h2 className="text-xl font-semibold mb-2">Ready to Analyze</h2>
                                    <p className="text-sm">Upload medical images to begin analysis</p>
                                </div>
                            </div>
                        ) : analysisResults.length === 0 && !isAnalyzing ? (
                            <div className="flex flex-col items-center justify-center h-full">
                                <img
                                    src={`${API_BASE}/${currentImage}`}
                                    alt="Current X-ray"
                                    className="max-w-md rounded-lg border border-zinc-700 mb-6"
                                />
                                <button
                                    onClick={runCompleteAnalysis}
                                    disabled={isAnalyzing}
                                    className="px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium flex items-center gap-2 disabled:opacity-50"
                                >
                                    {isAnalyzing ? (
                                        <>
                                            <Loader2 className="h-5 w-5 animate-spin" />
                                            Analyzing...
                                        </>
                                    ) : (
                                        '🔬 Run Complete Analysis'
                                    )}
                                </button>
                            </div>
                        ) : (
                            <div className="space-y-6">
                                {currentImage && (
                                    <div className="text-center mb-6">
                                        <h3 className="text-xs text-zinc-500 mb-2">Original Image</h3>
                                        <img
                                            src={`${API_BASE}/${currentImage}`}
                                            alt="Original X-ray"
                                            className="max-w-md rounded-lg border border-zinc-700 mx-auto"
                                        />
                                    </div>
                                )}

                                <h2 className="text-2xl font-semibold">Analysis Results</h2>

                                {analysisResults
                                    .filter(([toolName, _]: [string, any]) => {
                                        // Filter out pure utility tools
                                        const isUtility = toolName.includes('visualizer') || toolName.includes('image_visualizer');
                                        return !isUtility;
                                    })
                                    .map(([toolName, result]: [string, any], idx) => {
                                        const isClassification = toolName.includes('classification') || toolName.includes('classifier');
                                        const isSegmentation = toolName.includes('segmentation');
                                        const isReport = toolName.includes('report');
                                        const isGrounding = toolName.includes('grounding');
                                        const isExpert = toolName.includes('expert') || toolName.includes('vqa');

                                        return (
                                            <div key={idx} className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-6">
                                                <h3 className="text-lg font-semibold mb-4">
                                                    {toolName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                                </h3>

                                                {isClassification && (
                                                    <ClassificationResults
                                                        result={result}
                                                        idx={idx}
                                                        apiBase={API_BASE}
                                                    />
                                                )}

                                                {isSegmentation && (
                                                    <SegmentationResults
                                                        result={result}
                                                        idx={idx}
                                                        currentImage={currentImage}
                                                        apiBase={API_BASE}
                                                    />
                                                )}

                                                {isReport && (
                                                    <ReportResults
                                                        result={result}
                                                        idx={idx}
                                                    />
                                                )}

                                                {!isClassification && !isSegmentation && !isReport && (isExpert || isGrounding) && (
                                                    <div className="space-y-4">
                                                        {/* Display expert/grounding results nicely */}
                                                        <div className="text-sm text-zinc-300 whitespace-pre-wrap">
                                                            {result.result?.response ||
                                                                (typeof result.result === 'string' ? result.result : '') ||
                                                                (typeof result === 'string' ? result : '')}
                                                        </div>
                                                        {(result.result?.response || result.result) && (
                                                            <button
                                                                onClick={() => {
                                                                    const el = document.getElementById(`raw-${toolName}`);
                                                                    if (el) el.classList.toggle('hidden');
                                                                }}
                                                                className="text-xs text-zinc-500 hover:text-zinc-300"
                                                            >
                                                                Show/Hide raw data
                                                            </button>
                                                        )}
                                                        <pre id={`raw-${toolName}`} className="hidden bg-zinc-950 p-4 rounded-lg text-xs overflow-x-auto">
                                                            {JSON.stringify(result, null, 2)}
                                                        </pre>
                                                    </div>
                                                )}

                                                {!isClassification && !isSegmentation && !isReport && !isExpert && !isGrounding && (
                                                    <div className="text-sm text-zinc-300">
                                                        {typeof result === 'string' ? result : JSON.stringify(result, null, 2)}
                                                    </div>
                                                )}
                                            </div>
                                        );
                                    })}
                            </div>
                        )}
                    </div>

                    {/* Right: Chat */}
                    <div className="w-96 bg-zinc-900/50 border-l border-zinc-800 flex flex-col">
                        <div className="p-4 border-b border-zinc-800">
                            <h2 className="text-sm font-semibold text-zinc-400">Chat with AI</h2>
                        </div>

                        <div className="flex-1 overflow-y-auto p-4 space-y-3">
                            {messages.map((msg, idx) => (
                                <div key={idx} className={`${msg.role === 'user' ? 'text-right' : ''}`}>
                                    <div
                                        className={`inline-block px-4 py-2 rounded-lg text-sm max-w-full ${msg.role === 'user'
                                            ? 'bg-blue-600 text-white'
                                            : msg.role === 'system'
                                                ? 'bg-zinc-800 text-zinc-300'
                                                : 'bg-zinc-800 text-white'
                                            }`}
                                    >
                                        <div className="whitespace-pre-wrap break-words">
                                            {msg.content}
                                        </div>
                                    </div>
                                    <div className="text-xs text-zinc-600 mt-1">
                                        {msg.timestamp.toLocaleTimeString()}
                                    </div>
                                </div>
                            ))}
                            <div ref={messagesEndRef} />
                        </div>

                        <div className="p-4 border-t border-zinc-800">
                            <div className="flex gap-2">
                                <input
                                    type="text"
                                    value={inputMessage}
                                    onChange={(e) => setInputMessage(e.target.value)}
                                    onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                                    placeholder="Ask about the analysis..."
                                    className="flex-1 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm"
                                    disabled={isLoading}
                                />
                                <button
                                    onClick={sendMessage}
                                    disabled={isLoading}
                                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg disabled:opacity-50"
                                >
                                    {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : '→'}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Backend Logs - Collapsible at bottom */}
                {backendLogs.length > 0 && (
                    <div className="border-t border-zinc-800 bg-zinc-900">
                        <button
                            onClick={() => setShowLogs(!showLogs)}
                            className="w-full px-4 py-2 flex items-center justify-between hover:bg-zinc-800/50 transition-colors"
                        >
                            <div className="flex items-center gap-2">
                                <Terminal className="h-4 w-4 text-zinc-400" />
                                <span className="text-sm font-medium text-zinc-400">
                                    Backend Logs ({backendLogs.length})
                                </span>
                            </div>
                            {showLogs ? <ChevronDown className="h-4 w-4" /> : <ChevronUp className="h-4 w-4" />}
                        </button>
                        {showLogs && (
                            <div className="px-4 pb-4 max-h-48 overflow-y-auto">
                                <div className="bg-zinc-950 rounded-lg p-3 font-mono text-xs space-y-1">
                                    {backendLogs.map((log, idx) => (
                                        <div key={idx} className="text-zinc-400">{log}</div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
