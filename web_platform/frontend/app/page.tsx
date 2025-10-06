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
import { Bot, Loader2, Terminal, ChevronDown, ChevronUp } from 'lucide-react';
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

            // Fetch updated results after each message (tools might have been called)
            try {
                const resultsResponse = await axios.get(`${API_BASE}/api/analysis/${sessionId}`);
                if (resultsResponse.data.results && Object.keys(resultsResponse.data.results).length > 0) {
                    setAnalysisResults(Object.entries(resultsResponse.data.results));
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
                                        );
                                    })}
                            </div>
                        )}
                    </div>

                    {/* Right: Chat / Tools Panel */}
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
