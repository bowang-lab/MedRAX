'use client';

/**
 * NOTE: Zustand store is available in lib/store.ts for state management
 * To migrate state, replace useState with useAppStore, e.g.:
 * 
 * Before: const [sessionId, setSessionId] = useState(null);
 * After:  const { sessionId, setSessionId } = useAppStore();
 * 
 * This provides better state management, persistence, and performance.
 */

import { useState, useRef, useEffect } from 'react';
import { Bot, Loader2, Terminal, ChevronDown, ChevronUp, ChevronLeft, ChevronRight } from 'lucide-react';
import axios from 'axios';
import { useAppStore } from '../lib/store';
import Header from '../components/Header';
import PatientInfoForm from '../components/PatientInfoForm';
import PatientSidebar from '../components/PatientSidebar';
import ChatPanel from '../components/ChatPanel';
import ClassificationResults from '../components/ClassificationResults';
import SegmentationResults from '../components/SegmentationResults';
import ReportResults from '../components/ReportResults';
import GroundingResults from '../components/GroundingResults';
import VQAResults from '../components/VQAResults';
import ImageModal from '../components/ImageModal';
import ImageUploadZone from '../components/ImageUploadZone';
import AnalysisProgress from '../components/AnalysisProgress';
import PipelineVisualization from '../components/PipelineVisualization';
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

    // Collapsible states
    const [patientSidebarCollapsed, setPatientSidebarCollapsed] = useState(false);
    const [leftSidebarCollapsed, setLeftSidebarCollapsed] = useState(false);
    const [rightSidebarCollapsed, setRightSidebarCollapsed] = useState(true); // Start collapsed
    const [collapsedTools, setCollapsedTools] = useState<Set<number>>(new Set());

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

    // UI state
    const [rightSidebarMode, setRightSidebarMode] = useState<'chat' | 'tools'>('chat');
    const [pipelineSteps, setPipelineSteps] = useState<any[]>([]);
    const [modalImage, setModalImage] = useState<{ src: string, alt: string } | null>(null);

    const fileInputRef = useRef<HTMLInputElement>(null);

    const currentImage = uploadedImages[currentImageIndex] || null;

    // Load session history on mount
    useEffect(() => {
        setSessionHistory(getAllSessions());
        createSession();
    }, []);

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

    const clearChat = async () => {
        if (!sessionId) return;

        try {
            await axios.post(`${API_BASE}/api/sessions/${sessionId}/clear`);
            setMessages([]);
            setUploadedImages([]);
            setCurrentImageIndex(0);
            setAnalysisResults([]);
            console.log('✅ Chat cleared');

            // Add system message
            setMessages([{
                role: 'system',
                content: '🧹 Chat cleared. Session preserved. Upload new images to continue.',
                timestamp: new Date()
            }]);
        } catch (error: any) {
            console.error('Failed to clear chat:', error);
        }
    };

    const newThread = async () => {
        if (!sessionId) return;

        try {
            const response = await axios.post(`${API_BASE}/api/sessions/${sessionId}/new-thread`);
            setMessages([]);
            console.log('✅ New conversation thread started:', response.data.thread_id);

            // Add system message
            setMessages([{
                role: 'system',
                content: '🔄 New conversation started. Previous context cleared.',
                timestamp: new Date()
            }]);
        } catch (error: any) {
            console.error('Failed to start new thread:', error);
        }
    };

    const loadSession = async (savedSession: SessionData) => {
        // Check if session exists on backend
        try {
            await axios.get(`${API_BASE}/api/chat/${savedSession.sessionId}`);
            // Session exists, load it
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
        } catch (error) {
            console.error('Session not found on backend, creating new session with old data');
            // Session doesn't exist on backend, create a new one
            const newSessionId = await createSession();
            if (newSessionId) {
                // Load the old patient data
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
            }
        }
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

        // Auto-open chat sidebar on analysis
        if (rightSidebarCollapsed) {
            setRightSidebarCollapsed(false);
            setRightSidebarMode('chat'); // Switch to chat mode
        }

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
            // Use EventSource for Server-Sent Events streaming
            const eventSource = new EventSource(
                `${API_BASE}/api/chat/${sessionId}/stream?image_path=${encodeURIComponent(currentImage)}`
            );

            eventSource.addEventListener('status', (event) => {
                const data = JSON.parse(event.data);

                // Add message to chat
                setMessages(prev => [...prev, {
                    role: 'system',
                    content: data.message,
                    timestamp: new Date()
                }]);

                // Add to logs
                setBackendLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${data.message}`]);

                // Close connection when done
                if (data.type === 'done' || data.type === 'error') {
                    eventSource.close();

                    // Fetch final results
                    axios.get(`${API_BASE}/api/analysis/${sessionId}`)
                        .then(resultsResponse => {
                            console.log('✅ Analysis results received:', resultsResponse.data);
                            const results = resultsResponse.data.results || {};
                            console.log('📊 Results count:', Object.keys(results).length);
                            console.log('🔧 Tool names:', Object.keys(results));

                            setAnalysisResults(Object.entries(results));

                            // Format and add summary message to chat
                            let summaryMessage = '### 📊 Analysis Complete\n\n';

                            // Add classification results
                            if (results.chest_xray_classifier) {
                                const classData = results.chest_xray_classifier.result;
                                if (classData && typeof classData === 'object') {
                                    summaryMessage += '#### 🔬 Pathology Classification\n';
                                    const predictions = classData.predictions || classData;
                                    const topFindings = Object.entries(predictions)
                                        .sort((a: any, b: any) => b[1] - a[1])
                                        .slice(0, 5);
                                    topFindings.forEach(([name, score]: any) => {
                                        summaryMessage += `- **${name}:** ${(score * 100).toFixed(1)}%\n`;
                                    });
                                    summaryMessage += '\n';
                                }
                            }

                            // Add segmentation summary
                            if (results.chest_xray_segmentation) {
                                const segData = results.chest_xray_segmentation.result;
                                if (segData && typeof segData === 'object' && segData.organs) {
                                    summaryMessage += '#### 🫁 Anatomical Segmentation\n';
                                    summaryMessage += `Found ${Object.keys(segData.organs).length} anatomical structures\n\n`;
                                }
                            }

                            // Add report
                            if (results.chest_xray_report_generator) {
                                const report = results.chest_xray_report_generator.result;
                                if (typeof report === 'string') {
                                    summaryMessage += '#### 📝 Radiology Report\n\n';
                                    summaryMessage += report + '\n\n';
                                }
                            }

                            summaryMessage += '---\n\n';
                            summaryMessage += '*View detailed results in the panel on the left.*';

                            // Add the summary to chat
                            setMessages(prev => [...prev, {
                                role: 'assistant',
                                content: summaryMessage,
                                timestamp: new Date()
                            }]);

                            setIsAnalyzing(false);
                        })
                        .catch(err => {
                            console.error('❌ Failed to fetch results:', err);
                            setIsAnalyzing(false);
                        });
                }
            });

            eventSource.onerror = (error) => {
                console.error('EventSource error:', error);
                eventSource.close();
                setMessages(prev => [...prev, {
                    role: 'system',
                    content: '❌ Analysis stream connection failed',
                    timestamp: new Date()
                }]);
                setIsAnalyzing(false);
            };

        } catch (error: any) {
            console.error('Analysis failed:', error);
            setMessages(prev => [...prev, {
                role: 'system',
                content: `❌ Analysis failed: ${error.message}`,
                timestamp: new Date()
            }]);
            setIsAnalyzing(false);
        }
    };

    const sendMessage = async () => {
        if (!inputMessage.trim() || isLoading || !sessionId) return;

        // Auto-open chat sidebar on first message
        if (rightSidebarCollapsed) {
            setRightSidebarCollapsed(false);
            setRightSidebarMode('chat'); // Switch to chat mode
        }

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

            // Show tool execution if any tools were called
            if (response.data.tool_calls && response.data.tool_calls.length > 0) {
                const toolNames = response.data.tool_calls.map((tc: any) => tc.tool_name).join(', ');
                setBackendLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] 🔧 Tools used: ${toolNames}`]);

                // Add tool execution message
                setMessages(prev => [...prev, {
                    role: 'system',
                    content: `🔧 **Tool Execution:**\n${response.data.tool_calls.map((tc: any) => `✅ ${tc.tool_name}`).join('\n')}`,
                    timestamp: new Date()
                }]);
            }

            setMessages(prev => [...prev, {
                role: 'assistant',
                content: response.data.response,
                timestamp: new Date()
            }]);

            // Fetch updated results after each message (tools might have been called)
            try {
                const resultsResponse = await axios.get(`${API_BASE}/api/analysis/${sessionId}`);
                if (resultsResponse.data.results && Object.keys(resultsResponse.data.results).length > 0) {
                    setAnalysisResults(Object.entries(resultsResponse.data.results));
                    setBackendLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] 📊 Results updated: ${Object.keys(resultsResponse.data.results).length} tools`]);
                }
            } catch (err) {
                console.error('Failed to fetch results:', err);
            }
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

    // Toggle tool collapse
    const toggleToolCollapse = (index: number) => {
        const newCollapsed = new Set(collapsedTools);
        if (newCollapsed.has(index)) {
            newCollapsed.delete(index);
        } else {
            newCollapsed.add(index);
        }
        setCollapsedTools(newCollapsed);
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
                onSelectSession={async (id) => {
                    const session = getSession(id);
                    if (session) await loadSession(session);
                }}
                onNewPatient={startNewPatient}
                collapsed={patientSidebarCollapsed}
                onToggleCollapse={() => setPatientSidebarCollapsed(!patientSidebarCollapsed)}
            />

            {/* Patient Sidebar Expand Button (when collapsed) */}
            {patientSidebarCollapsed && (
                <button
                    onClick={() => setPatientSidebarCollapsed(false)}
                    className="fixed left-0 top-1/2 -translate-y-1/2 z-20 bg-zinc-800 hover:bg-zinc-700 text-white p-2 rounded-r-lg border border-l-0 border-zinc-700 transition-all duration-200 hover:scale-110 shadow-lg"
                >
                    <ChevronRight className="h-4 w-4" />
                </button>
            )}

            {/* Main Content */}
            <div className="flex-1 flex flex-col">
                {/* Header */}
                <Header
                    sessionId={sessionId}
                    patientInfo={patientInfo}
                    isAnalyzing={isAnalyzing}
                    showPatientForm={showPatientForm}
                    onTogglePatientForm={() => setShowPatientForm(!showPatientForm)}
                />

                {/* Patient Info Form */}
                {showPatientForm && (
                    <PatientInfoForm
                        patientInfo={patientInfo}
                        onChange={setPatientInfo}
                        onClose={() => setShowPatientForm(false)}
                    />
                )}

                {/* Main Content Area */}
                <div className="flex-1 flex flex-col overflow-hidden">
                    <div className="flex-1 flex overflow-hidden relative">
                        {/* Left: Image Upload Sidebar - Only show when images exist */}
                        {uploadedImages.length > 0 && (
                            <div className={`transition-all duration-300 ease-in-out bg-gradient-to-br from-zinc-900/80 to-zinc-900/50 backdrop-blur-sm border-r border-zinc-800/50 flex flex-col ${leftSidebarCollapsed ? 'w-0' : 'w-80'} overflow-hidden`}>
                                <div className="p-4 border-b border-zinc-800/50 flex items-center justify-between">
                                    <h2 className="text-sm font-semibold text-zinc-300 flex items-center gap-2">
                                        <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></div>
                                        Medical Images
                                    </h2>
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
                                                className={`relative cursor-pointer rounded-xl overflow-hidden border-2 transition-all duration-200 hover:scale-[1.02] ${idx === currentImageIndex
                                                    ? 'border-blue-500 shadow-lg shadow-blue-500/20'
                                                    : 'border-zinc-800/50 hover:border-zinc-700'
                                                    }`}
                                            >
                                                <img
                                                    src={`${API_BASE}/${img}`}
                                                    alt={`Upload ${idx + 1}`}
                                                    className="w-full h-32 object-cover"
                                                />
                                                {idx === currentImageIndex && (
                                                    <div className="absolute top-2 right-2 px-2 py-1 bg-blue-500 text-white text-xs rounded-full font-semibold">
                                                        Active
                                                    </div>
                                                )}
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
                        )}

                        {/* Left Sidebar Collapse Button - Only show when images exist */}
                        {uploadedImages.length > 0 && (
                            <button
                                onClick={() => setLeftSidebarCollapsed(!leftSidebarCollapsed)}
                                className="absolute left-0 top-1/2 -translate-y-1/2 z-10 bg-zinc-800 hover:bg-zinc-700 text-white p-2 rounded-r-lg border border-l-0 border-zinc-700 transition-all duration-200 hover:scale-110 shadow-lg"
                                style={{ left: leftSidebarCollapsed ? '0' : '320px' }}
                            >
                                {leftSidebarCollapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
                            </button>
                        )}

                        {/* Center: Results or Upload */}
                        <div className="flex-1 overflow-y-auto p-6 bg-gradient-to-b from-zinc-950 to-zinc-900">
                            {uploadedImages.length === 0 ? (
                                <div className="flex items-center justify-center h-full">
                                    <div className="text-center max-w-2xl mx-auto">
                                        <div className="relative mb-8">
                                            <div className="absolute inset-0 blur-3xl bg-blue-500/20 rounded-full"></div>
                                            <Bot className="relative h-24 w-24 mx-auto text-blue-400" />
                                        </div>
                                        <h2 className="text-3xl font-bold mb-3 bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent">
                                            Ready to Analyze
                                        </h2>
                                        <p className="text-zinc-400 mb-8">Upload medical images to begin AI-powered analysis</p>

                                        {/* Large centered upload zone */}
                                        <div
                                            onDragEnter={(e) => { e.preventDefault(); setDragActive(true); }}
                                            onDragLeave={(e) => { e.preventDefault(); setDragActive(false); }}
                                            onDragOver={(e) => e.preventDefault()}
                                            onDrop={handleDrop}
                                            onClick={() => fileInputRef.current?.click()}
                                            className={`relative cursor-pointer border-2 border-dashed rounded-2xl p-16 transition-all duration-300 ${dragActive
                                                ? 'border-blue-500 bg-blue-500/10 scale-105'
                                                : 'border-zinc-700 hover:border-zinc-600 hover:bg-zinc-800/30'
                                                }`}
                                        >
                                            <div className="flex flex-col items-center gap-4">
                                                <div className="p-6 rounded-full bg-zinc-800/50">
                                                    <svg className="h-16 w-16 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                                                    </svg>
                                                </div>
                                                <div>
                                                    <h3 className="text-xl font-semibold mb-2 text-white">Drop X-ray images here</h3>
                                                    <p className="text-zinc-400 text-sm">or click to browse</p>
                                                </div>
                                                <div className="flex items-center gap-2 text-xs text-zinc-500">
                                                    <span>Supports DICOM, JPG, PNG</span>
                                                </div>
                                            </div>
                                        </div>
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
                            ) : analysisResults.length === 0 && !isAnalyzing ? (
                                <div className="flex flex-col items-center justify-center h-full">
                                    <div className="relative mb-8 group">
                                        <div className="absolute inset-0 bg-blue-500/10 blur-2xl rounded-lg group-hover:bg-blue-500/20 transition-all"></div>
                                        <img
                                            src={`${API_BASE}/${currentImage}`}
                                            alt="Current X-ray"
                                            className="relative max-w-md rounded-xl border-2 border-zinc-800/50 shadow-2xl group-hover:border-zinc-700/50 transition-all"
                                        />
                                    </div>
                                    <button
                                        onClick={runCompleteAnalysis}
                                        disabled={isAnalyzing}
                                        className="px-8 py-4 bg-gradient-to-r from-blue-600 to-emerald-600 hover:from-blue-500 hover:to-emerald-500 rounded-xl font-semibold flex items-center gap-3 disabled:opacity-50 shadow-lg shadow-blue-500/25 hover:shadow-xl hover:shadow-blue-500/40 transition-all duration-200 hover:scale-105"
                                    >
                                        {isAnalyzing ? (
                                            <>
                                                <Loader2 className="h-5 w-5 animate-spin" />
                                                Analyzing...
                                            </>
                                        ) : (
                                            <>
                                                🔬 Run Complete Analysis
                                            </>
                                        )}
                                    </button>
                                </div>
                            ) : (
                                <div className="space-y-6">
                                    {currentImage && (
                                        <div className="text-center mb-8">
                                            <h3 className="text-xs text-zinc-400 mb-3 uppercase tracking-wider font-semibold">Original Image</h3>
                                            <div className="relative inline-block">
                                                <div className="absolute inset-0 bg-blue-500/5 blur-xl rounded-xl"></div>
                                                <img
                                                    src={`${API_BASE}/${currentImage}`}
                                                    alt="Original X-ray"
                                                    className="relative max-w-md rounded-xl border-2 border-zinc-800/50 mx-auto shadow-xl"
                                                />
                                            </div>
                                        </div>
                                    )}

                                    <div className="flex items-center gap-3 mb-6">
                                        <div className="flex-1 h-px bg-gradient-to-r from-transparent via-zinc-700 to-transparent"></div>
                                        <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent">
                                            Analysis Results
                                        </h2>
                                        <div className="flex-1 h-px bg-gradient-to-r from-transparent via-zinc-700 to-transparent"></div>
                                    </div>

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

                                            const isCollapsed = collapsedTools.has(idx);

                                            return (
                                                <div key={idx} className="bg-gradient-to-br from-zinc-900/80 to-zinc-900/50 backdrop-blur-sm border border-zinc-800/50 rounded-xl overflow-hidden shadow-lg hover:shadow-xl transition-all duration-300 hover:border-zinc-700/50">
                                                    <div
                                                        onClick={() => toggleToolCollapse(idx)}
                                                        className="flex items-center justify-between p-6 cursor-pointer hover:bg-zinc-800/30 transition-colors"
                                                    >
                                                        <h3 className="text-lg font-semibold flex items-center gap-3">
                                                            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
                                                            {toolName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                                        </h3>
                                                        <button className="p-1 hover:bg-zinc-700 rounded-lg transition-colors">
                                                            {isCollapsed ? <ChevronDown className="h-5 w-5" /> : <ChevronUp className="h-5 w-5" />}
                                                        </button>
                                                    </div>

                                                    {!isCollapsed && (
                                                        <div className="px-6 pb-6"  >

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

                                                            {isGrounding && (
                                                                <GroundingResults
                                                                    result={result}
                                                                    idx={idx}
                                                                    apiBase={API_BASE}
                                                                />
                                                            )}

                                                            {isExpert && (
                                                                <VQAResults
                                                                    result={result}
                                                                    idx={idx}
                                                                />
                                                            )}

                                                            {!isClassification && !isSegmentation && !isReport && !isExpert && !isGrounding && (
                                                                <div className="text-sm text-zinc-300">
                                                                    {typeof result === 'string' ? result : JSON.stringify(result, null, 2)}
                                                                </div>
                                                            )}
                                                        </div>
                                                    )}
                                                </div>
                                            );
                                        })}
                                </div>
                            )}
                        </div>

                        {/* Right: Chat / Tools Panel */}
                        <div className={`h-full transition-all duration-300 ease-in-out ${rightSidebarCollapsed ? 'w-0' : 'w-96'} overflow-hidden`}>
                            <ChatPanel
                                sessionId={sessionId}
                                messages={messages}
                                inputMessage={inputMessage}
                                isLoading={isLoading}
                                rightSidebarMode={rightSidebarMode}
                                apiBase={API_BASE}
                                onInputChange={setInputMessage}
                                onSendMessage={sendMessage}
                                onClearChat={clearChat}
                                onNewThread={newThread}
                                onSetMode={setRightSidebarMode}
                                onImageClick={(src, alt) => setModalImage({ src, alt })}
                            />
                        </div>

                        {/* Right Sidebar Collapse Button */}
                        <button
                            onClick={() => setRightSidebarCollapsed(!rightSidebarCollapsed)}
                            className="absolute right-0 top-1/2 -translate-y-1/2 z-10 bg-zinc-800 hover:bg-zinc-700 text-white p-2 rounded-l-lg border border-r-0 border-zinc-700 transition-all duration-200 hover:scale-110 shadow-lg"
                            style={{ right: rightSidebarCollapsed ? '0' : '384px' }}
                        >
                            {rightSidebarCollapsed ? <ChevronLeft className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                        </button>
                    </div>

                    {/* Bottom Chat Input - Only show when chat sidebar is collapsed/in tools mode */}
                    {(rightSidebarCollapsed || rightSidebarMode === 'tools') && (
                        <div
                            className="border-t border-zinc-800/50 bg-gradient-to-r from-zinc-900/95 to-zinc-900/80 backdrop-blur-sm transition-all duration-300"
                            style={{
                                marginRight: rightSidebarMode === 'tools' && !rightSidebarCollapsed ? '384px' : '0'
                            }}
                        >
                            <div className="max-w-4xl mx-auto p-4">
                                <div className="flex gap-3 items-center">
                                    <input
                                        type="text"
                                        value={inputMessage}
                                        onChange={(e) => setInputMessage(e.target.value)}
                                        onKeyPress={(e) => e.key === 'Enter' && !isLoading && sendMessage()}
                                        placeholder="Ask about the analysis..."
                                        className="flex-1 px-5 py-4 bg-zinc-800/50 border border-zinc-700/50 rounded-2xl text-sm placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-transparent transition-all backdrop-blur-sm"
                                        disabled={isLoading || !sessionId}
                                    />
                                    <button
                                        onClick={sendMessage}
                                        disabled={isLoading || !sessionId || !inputMessage.trim()}
                                        className="px-6 py-4 bg-gradient-to-r from-blue-600 to-emerald-600 hover:from-blue-500 hover:to-emerald-500 rounded-2xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 hover:scale-105 shadow-lg shadow-blue-500/25 hover:shadow-xl hover:shadow-blue-500/40 font-semibold"
                                    >
                                        {isLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : <span className="text-xl">→</span>}
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Image Modal */}
                {modalImage && (
                    <ImageModal
                        src={modalImage.src}
                        alt={modalImage.alt}
                        onClose={() => setModalImage(null)}
                    />
                )}

                {/* Pipeline Visualization */}
                <PipelineVisualization steps={pipelineSteps} isActive={isAnalyzing} />

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
