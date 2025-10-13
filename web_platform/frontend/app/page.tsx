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
import { Bot, Loader2, Terminal, ChevronDown, ChevronUp, ChevronLeft, ChevronRight, Send, Settings, MessageSquare, Image, Plus } from 'lucide-react';
import axios from 'axios';
import { useAppStore } from '../lib/store';
import Header from '../components/layouts/Header';
import PatientInfoForm from '../components/features/PatientInfoForm';
import PatientSidebar from '../components/features/PatientSidebar';
import ChatSidebar from '../components/features/ChatSidebar';
import ToolOutputPanel from '../components/features/ToolOutputPanel';
import ToolsPanel from '../components/features/ToolsPanel';
import ImageGallery from '../components/features/ImageGallery';
import ImageModal from '../components/ui/ImageModal';
import ImageUploadZone from '../components/ui/ImageUploadZone';
import AnalysisProgress from '../components/ui/AnalysisProgress';
import PipelineVisualization from '../components/ui/PipelineVisualization';
import MessageRenderer from '../components/ui/MessageRenderer';
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
    // Generate initial userId only once on mount
    const [initialUserIdGenerated] = useState(() => `user-${Date.now()}`);

    // Core state - Multi-chat architecture
    const [userId, setUserId] = useState<string>(initialUserIdGenerated);
    const [currentChatId, setCurrentChatId] = useState<string | null>(null);
    const [chats, setChats] = useState<any[]>([]);

    const [messages, setMessages] = useState<Message[]>([]);
    const [inputMessage, setInputMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [uploadedImages, setUploadedImages] = useState<string[]>([]);
    const [currentImageIndex, setCurrentImageIndex] = useState(0);
    const [dragActive, setDragActive] = useState(false);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [analysisResults, setAnalysisResults] = useState<any[]>([]);
    const [toolHistory, setToolHistory] = useState<any[]>([]);  // Full tool execution history
    const [toolFilterMode, setToolFilterMode] = useState<'latest' | 'all' | 'request'>('latest');  // Filtering mode
    const [currentRequestId, setCurrentRequestId] = useState<string | null>(null);  // Track latest request ID

    // Collapsible states - Updated for new layout
    const [patientSidebarCollapsed, setPatientSidebarCollapsed] = useState(false);
    const [chatSidebarCollapsed, setChatSidebarCollapsed] = useState(false);
    const [imageSidebarCollapsed, setImageSidebarCollapsed] = useState(false);
    const [toolOutputCollapsed, setToolOutputCollapsed] = useState(true); // Start collapsed, auto-expand when results arrive

    // Sidebar width states (in pixels)
    const [patientSidebarWidth, setPatientSidebarWidth] = useState(320);
    const [chatSidebarWidth, setChatSidebarWidth] = useState(280);
    const [imageSidebarWidth, setImageSidebarWidth] = useState(300);
    const [toolOutputWidth, setToolOutputWidth] = useState(400);

    // Resize state
    const [isResizing, setIsResizing] = useState<string | null>(null);

    // Right sidebar mode: 'results' (tool outputs) or 'management' (tool loading/management)
    const [rightSidebarMode, setRightSidebarMode] = useState<'results' | 'management'>('results');

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
    const [pipelineSteps, setPipelineSteps] = useState<any[]>([]);
    const [modalImage, setModalImage] = useState<{ src: string, alt: string } | null>(null);

    const fileInputRef = useRef<HTMLInputElement>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const currentImage = uploadedImages[currentImageIndex] || null;

    // Load sidebar widths from localStorage on mount
    useEffect(() => {
        const savedPatientWidth = localStorage.getItem('patientSidebarWidth');
        const savedChatWidth = localStorage.getItem('chatSidebarWidth');
        const savedImageWidth = localStorage.getItem('imageSidebarWidth');
        const savedToolOutputWidth = localStorage.getItem('toolOutputWidth');

        if (savedPatientWidth) setPatientSidebarWidth(parseInt(savedPatientWidth));
        if (savedChatWidth) setChatSidebarWidth(parseInt(savedChatWidth));
        if (savedImageWidth) setImageSidebarWidth(parseInt(savedImageWidth));
        if (savedToolOutputWidth) setToolOutputWidth(parseInt(savedToolOutputWidth));
    }, []);

    // Auto-scroll to bottom of messages
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // Track if chats have been loaded to prevent double loading
    const chatsLoadedRef = useRef(false);
    const previousUserIdRef = useRef<string | null>(null);

    // Load chats on mount and when userId changes
    useEffect(() => {
        // Reset loaded flag if userId actually changed
        if (previousUserIdRef.current !== userId) {
            chatsLoadedRef.current = false;
            previousUserIdRef.current = userId;
        }

        // Prevent double loading (React StrictMode issue)
        if (chatsLoadedRef.current) {
            console.log('⏭️  Skipping duplicate chat load (already loaded)');
            return;
        }
        chatsLoadedRef.current = true;

        console.log('📋 Loading chats for userId:', userId);
        setSessionHistory(getAllSessions());

        // Clear current state when switching users
        setMessages([]);
        setUploadedImages([]);
        setCurrentImageIndex(0);
        setAnalysisResults([]);
        setCurrentChatId(null);
        setChats([]);

        // Load chats for the current user
        loadUserChats();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [userId]);

    // Load chats for current user
    const loadUserChats = async () => {
        try {
            const response = await axios.get(`${API_BASE}/api/users/${userId}/chats`);
            setChats(response.data.chats);

            // If no chats exist, create first one
            if (response.data.chats.length === 0) {
                console.log('📝 No chats found, creating first chat...');
                await createNewChat();
            } else {
                // Select the first chat
                console.log('📂 Loading existing chat:', response.data.chats[0].chat_id);
                await selectChat(response.data.chats[0].chat_id);
            }
        } catch (error: any) {
            console.error('Failed to load chats:', error);
            // Only create a new chat if it's a connection error (user/chat doesn't exist yet)
            // Don't create duplicate chats on other errors
            if (error.response?.status === 404) {
                console.log('📝 User has no chats yet, creating first chat...');
                await createNewChat();
            } else {
                console.error('⚠️ Chat loading failed with unexpected error');
                // Show error message but don't create duplicate chats
                setMessages([{
                    role: 'system',
                    content: '⚠️ Failed to load chats. Please refresh the page.',
                    timestamp: new Date()
                }]);
            }
        }
    };

    // Create a new chat
    const createNewChat = async (chatName?: string) => {
        try {
            const response = await axios.post(`${API_BASE}/api/users/${userId}/chats`, null, {
                params: { chat_name: chatName }
            });

            const newChat = response.data;
            setChats(prev => [newChat, ...prev]);
            setCurrentChatId(newChat.chat_id);

            // Reset state for new chat
            setMessages([]);
            setUploadedImages([]);
            setCurrentImageIndex(0);
            setAnalysisResults([]);

            console.log('✅ New chat created:', newChat.chat_id);

            setMessages([{
                role: 'system',
                content: '👋 New chat started. Upload medical images to begin analysis.',
                timestamp: new Date()
            }]);

            return newChat.chat_id;
        } catch (error) {
            console.error('Failed to create chat:', error);
            return null;
        }
    };

    // Select a chat
    const selectChat = async (chatId: string) => {
        try {
            setCurrentChatId(chatId);

            // Fetch chat details
            const response = await axios.get(`${API_BASE}/api/users/${userId}/chats/${chatId}`);
            const chatData = response.data;

            // Update state with chat data
            setUploadedImages(chatData.uploaded_images || []);
            setCurrentImageIndex(0);

            // Load message history from backend
            if (chatData.message_history && chatData.message_history.length > 0) {
                const loadedMessages = chatData.message_history.map((msg: any) => ({
                    role: msg.role,
                    content: msg.content,
                    timestamp: new Date(msg.timestamp)
                }));
                setMessages(loadedMessages);
            } else {
                // No history, show welcome message
                setMessages([{
                    role: 'system',
                    content: `📂 Chat loaded: ${chatData.metadata.name}`,
                    timestamp: new Date()
                }]);
            }

            // Load analysis results if available
            if (chatData.has_results) {
                const resultsResponse = await axios.get(`${API_BASE}/api/analysis/${chatId}`);
                setAnalysisResults(Object.entries(resultsResponse.data.results));
            } else {
                setAnalysisResults([]);
            }

            console.log('✅ Chat selected:', chatId);
        } catch (error) {
            console.error('Failed to load chat:', error);
        }
    };

    // Delete a chat
    const deleteChat = async (chatId: string) => {
        try {
            await axios.delete(`${API_BASE}/api/users/${userId}/chats/${chatId}`);
            setChats(prev => prev.filter(c => c.chat_id !== chatId));

            // If deleting current chat, switch to another or create new
            if (chatId === currentChatId) {
                const remainingChats = chats.filter(c => c.chat_id !== chatId);
                if (remainingChats.length > 0) {
                    await selectChat(remainingChats[0].chat_id);
                } else {
                    await createNewChat();
                }
            }

            console.log('✅ Chat deleted:', chatId);
        } catch (error) {
            console.error('Failed to delete chat:', error);
        }
    };

    // Save session to history when there are changes
    useEffect(() => {
        if (userId && (uploadedImages.length > 0 || messages.length > 0 || patientInfo.name)) {
            const sessionData: SessionData = {
                sessionId: userId, // Use userId as the session identifier
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
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [userId, uploadedImages.length, analysisResults.length, messages.length, patientInfo.name, patientInfo.age, patientInfo.gender, patientInfo.notes]);

    // Legacy functions removed - functionality replaced by:
    // - createSession() → createNewChat() (creates chat under current userId)
    // - clearChat() → createNewChat() (new chat = fresh context)
    // - newThread() → createNewChat() (new chat = new conversation)

    const loadSession = async (savedSession: SessionData) => {
        console.log('Loading patient session:', savedSession);

        // Set the user ID to the saved session's ID (use session ID as user ID)
        const sessionUserId = savedSession.sessionId;
        setUserId(sessionUserId);

        // Load patient info
        setPatientInfo({
            name: savedSession.patientName,
            age: savedSession.patientAge,
            gender: savedSession.patientGender,
            notes: savedSession.patientNotes
        });

        // The useEffect will trigger and load chats for this user
        console.log('✅ Patient loaded, fetching chats for userId:', sessionUserId);
    };

    const startNewPatient = async () => {
        // Clear all current state
        setMessages([]);
        setUploadedImages([]);
        setCurrentImageIndex(0);
        setAnalysisResults([]);
        setShowPatientForm(false);
        setPatientInfo({ name: '', age: '', gender: '', notes: '' });

        // Create a new user ID for the new patient
        const newUserId = `user-${Date.now()}`;
        setUserId(newUserId);

        // This will trigger the useEffect to load/create chats for the new user
        console.log('✅ New patient started with userId:', newUserId);

        // Show welcome message
        setMessages([{
            role: 'system',
            content: `🏥 New patient case started. Upload images to begin analysis.`,
            timestamp: new Date()
        }]);
    };

    const uploadFile = async (file: File) => {
        if (!currentChatId) {
            console.error('No active chat');
            return;
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
                `${API_BASE}/api/users/${userId}/chats/${currentChatId}/upload`,
                formData,
                { headers: { 'Content-Type': 'multipart/form-data' } }
            );

            setUploadedImages(prev => {
                const newImages = [...prev, response.data.display_path];
                setCurrentImageIndex(newImages.length - 1);
                return newImages;
            });

            // Update chat list with new image count
            setChats(prev => prev.map(chat =>
                chat.chat_id === currentChatId
                    ? { ...chat, image_count: response.data.total_images }
                    : chat
            ));

            setMessages(prev => {
                const newMessages = [...prev];
                newMessages[newMessages.length - 1] = {
                    role: 'system',
                    content: `✅ ${file.name} uploaded successfully! (${response.data.total_images} images total)`,
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
        if (!currentChatId || uploadedImages.length === 0 || isAnalyzing) return;

        // Auto-open tool output panel to show results
        if (toolOutputCollapsed) {
            setToolOutputCollapsed(false);
        }

        setIsAnalyzing(true);

        // Show which images are being analyzed
        const imageCountMsg = uploadedImages.length === 1
            ? '1 image'
            : `${uploadedImages.length} images`;

        setMessages(prev => [...prev, {
            role: 'system',
            content: `🤖 Starting comprehensive AI analysis of **${imageCountMsg}**...\n\n📊 This will analyze ALL uploaded images together and provide:\n- Pathology classification\n- Anatomical segmentation\n- Detailed radiology report\n- Clinical recommendations`,
            timestamp: new Date()
        }]);

        // Clear previous logs
        setBackendLogs([]);
        setBackendLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Starting analysis of ${imageCountMsg}...`]);

        try {
            // Use chat-specific streaming endpoint (analyzes ALL images in the chat)
            const streamUrl = `${API_BASE}/api/users/${userId}/chats/${currentChatId}/stream`;

            const eventSource = new EventSource(streamUrl);

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

                    // Fetch final results (both latest and history)
                    const resultsUrl = `${API_BASE}/api/users/${userId}/chats/${currentChatId}/results`;
                    const historyUrl = `${API_BASE}/api/users/${userId}/chats/${currentChatId}/tool-history`;

                    // Fetch both latest results and full history
                    Promise.all([
                        axios.get(resultsUrl),
                        axios.get(historyUrl)
                    ]).then(([resultsResponse, historyResponse]) => {
                        console.log('✅ Analysis results received:', resultsResponse.data);
                        const results = resultsResponse.data.results || {};
                        console.log('📊 Results count:', Object.keys(results).length);
                        console.log('🔧 Tool names:', Object.keys(results));

                        setAnalysisResults(Object.entries(results));

                        // Store full history
                        const history = historyResponse.data.history || [];
                        setToolHistory(history);
                        console.log('📜 Tool history loaded:', history.length, 'executions');

                        // Track the latest request ID
                        if (history.length > 0) {
                            setCurrentRequestId(history[history.length - 1].request_id);
                        }

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
        if (!inputMessage.trim() || isLoading || !currentChatId) return;

        const userMessage = inputMessage;
        setInputMessage('');
        setIsLoading(true);

        setMessages(prev => [...prev, {
            role: 'user',
            content: userMessage,
            timestamp: new Date()
        }]);

        try {
            const response = await axios.post(`${API_BASE}/api/users/${userId}/chats/${currentChatId}/messages`, {
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

            // Update chat list with new message count
            setChats(prev => prev.map(chat =>
                chat.chat_id === currentChatId
                    ? { ...chat, message_count: chat.message_count + 2, last_access: new Date().toISOString() }
                    : chat
            ));

            // Fetch updated results after each message (tools might have been called)
            try {
                const resultsResponse = await axios.get(`${API_BASE}/api/users/${userId}/chats/${currentChatId}/results`);
                if (resultsResponse.data.results && Object.keys(resultsResponse.data.results).length > 0) {
                    setAnalysisResults(Object.entries(resultsResponse.data.results));
                    setBackendLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] 📊 Results updated: ${Object.keys(resultsResponse.data.results).length} tools`]);

                    // Auto-expand tool output panel if collapsed
                    if (toolOutputCollapsed) {
                        setToolOutputCollapsed(false);
                    }
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

    // Sidebar resize handlers
    const MIN_SIDEBAR_WIDTH = 200;
    const MAX_SIDEBAR_WIDTH = 600;

    const startResize = (sidebar: string) => (e: React.MouseEvent) => {
        e.preventDefault();
        setIsResizing(sidebar);
    };

    useEffect(() => {
        if (!isResizing) return;

        const handleMouseMove = (e: MouseEvent) => {
            if (!isResizing) return;

            if (isResizing === 'patient') {
                const newWidth = Math.max(MIN_SIDEBAR_WIDTH, Math.min(MAX_SIDEBAR_WIDTH, e.clientX));
                setPatientSidebarWidth(newWidth);
                localStorage.setItem('patientSidebarWidth', newWidth.toString());
            } else if (isResizing === 'chat') {
                const patientWidth = patientSidebarCollapsed ? 0 : patientSidebarWidth;
                const newWidth = Math.max(MIN_SIDEBAR_WIDTH, Math.min(MAX_SIDEBAR_WIDTH, e.clientX - patientWidth));
                setChatSidebarWidth(newWidth);
                localStorage.setItem('chatSidebarWidth', newWidth.toString());
            } else if (isResizing === 'image') {
                const patientWidth = patientSidebarCollapsed ? 0 : patientSidebarWidth;
                const chatWidth = chatSidebarCollapsed ? 0 : chatSidebarWidth;
                const newWidth = Math.max(MIN_SIDEBAR_WIDTH, Math.min(MAX_SIDEBAR_WIDTH, e.clientX - patientWidth - chatWidth));
                setImageSidebarWidth(newWidth);
                localStorage.setItem('imageSidebarWidth', newWidth.toString());
            } else if (isResizing === 'toolOutput') {
                const newWidth = Math.max(MIN_SIDEBAR_WIDTH, Math.min(MAX_SIDEBAR_WIDTH, window.innerWidth - e.clientX));
                setToolOutputWidth(newWidth);
                localStorage.setItem('toolOutputWidth', newWidth.toString());
            }
        };

        const handleMouseUp = () => {
            setIsResizing(null);
        };

        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);

        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
        };
    }, [isResizing, patientSidebarCollapsed, chatSidebarCollapsed, patientSidebarWidth, chatSidebarWidth]);

    return (
        <div className="h-screen flex bg-zinc-950 text-white" style={{ cursor: isResizing ? 'col-resize' : 'default' }}>
            {/* Analysis Progress Overlay */}
            <AnalysisProgress isAnalyzing={isAnalyzing} />

            {/* Left: Patient History Sidebar */}
            {!patientSidebarCollapsed && (
                <div className="relative flex" style={{ width: `${patientSidebarWidth}px` }}>
                    <PatientSidebar
                        sessions={sessionHistory.map((s, idx) => ({
                            sessionId: s.sessionId,
                            patientName: s.patientName,
                            patientAge: s.patientAge,
                            timestamp: new Date(s.timestamp),
                            imageCount: s.imageCount,
                            isActive: s.sessionId === userId // Check against userId, not sessionId
                        }))}
                        onSelectSession={async (id) => {
                            const session = getSession(id);
                            if (session) await loadSession(session);
                        }}
                        onNewPatient={startNewPatient}
                        collapsed={false}
                        onToggleCollapse={() => setPatientSidebarCollapsed(!patientSidebarCollapsed)}
                    />
                    {/* Resize handle */}
                    <div
                        className="absolute right-0 top-0 bottom-0 w-1 hover:w-2 bg-gradient-to-b from-blue-500/20 to-emerald-500/20 hover:from-blue-500/50 hover:to-emerald-500/50 cursor-col-resize transition-all z-40"
                        onMouseDown={startResize('patient')}
                        title="Drag to resize"
                    />
                </div>
            )}

            {/* Patient Sidebar Expand Button (when collapsed) */}
            {patientSidebarCollapsed && (
                <button
                    onClick={() => setPatientSidebarCollapsed(false)}
                    className="fixed left-0 top-1/2 -translate-y-1/2 z-30 bg-gradient-to-r from-blue-600 to-emerald-600 hover:from-blue-500 hover:to-emerald-500 text-white p-3 rounded-r-xl border border-l-0 border-blue-500/50 transition-all duration-200 hover:scale-110 shadow-xl shadow-blue-500/50 hover:shadow-2xl hover:shadow-blue-500/60"
                    title="Show Patients"
                >
                    <ChevronRight className="h-5 w-5" />
                </button>
            )}

            {/* Chat Sidebar Expand Button (when collapsed) */}
            {chatSidebarCollapsed && (
                <button
                    onClick={() => setChatSidebarCollapsed(false)}
                    className="fixed top-1/2 -translate-y-1/2 z-30 bg-gradient-to-r from-emerald-600 to-blue-600 hover:from-emerald-500 hover:to-blue-500 text-white p-3 rounded-r-xl border border-l-0 border-emerald-500/50 transition-all duration-200 hover:scale-110 shadow-xl shadow-emerald-500/50 hover:shadow-2xl shadow-emerald-500/60"
                    style={{
                        left: `${patientSidebarCollapsed ? 0 : patientSidebarWidth}px`,
                        transform: patientSidebarCollapsed ? 'translate(0, 3rem)' : 'translateY(-50%)'
                    }}
                    title="Show Conversations"
                >
                    <ChevronRight className="h-5 w-5" />
                </button>
            )}

            {/* Chat Sidebar */}
            {!chatSidebarCollapsed && (
                <div className="relative flex" style={{ width: `${chatSidebarWidth}px` }}>
                    <ChatSidebar
                        userId={userId}
                        currentChatId={currentChatId}
                        chats={chats}
                        collapsed={false}
                        onSelectChat={selectChat}
                        onNewChat={createNewChat}
                        onDeleteChat={deleteChat}
                        onToggleCollapse={() => setChatSidebarCollapsed(!chatSidebarCollapsed)}
                    />
                    {/* Resize handle */}
                    <div
                        className="absolute right-0 top-0 bottom-0 w-1 hover:w-2 bg-gradient-to-b from-emerald-500/20 to-blue-500/20 hover:from-emerald-500/50 hover:to-blue-500/50 cursor-col-resize transition-all z-40"
                        onMouseDown={startResize('chat')}
                        title="Drag to resize"
                    />
                </div>
            )}

            {/* Main Content */}
            <div className="flex-1 flex flex-col">
                {/* Header */}
                <Header
                    sessionId={currentChatId}  // Note: renamed to chatId would be better but keeping for component compatibility
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
                    {/* Left: Image Gallery Sidebar - Show when images exist */}
                    {uploadedImages.length > 0 && !imageSidebarCollapsed && (
                        <div className="relative flex" style={{ width: `${imageSidebarWidth}px` }}>
                            <div className="flex-1 transition-all duration-300 ease-in-out bg-gradient-to-b from-zinc-900 via-zinc-900/95 to-zinc-900/90 border-r border-zinc-800/50 flex flex-col shadow-xl overflow-hidden">
                                {/* Header with gradient accent */}
                                <div className="relative p-4 border-b border-zinc-800/50 bg-gradient-to-r from-purple-900/20 via-zinc-900/50 to-pink-900/20">
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-2">
                                            <div className="p-2 rounded-lg bg-gradient-to-br from-purple-500/10 to-pink-500/10 border border-purple-500/20">
                                                <Image className="h-4 w-4 text-purple-400" />
                                            </div>
                                            <div>
                                                <h2 className="text-sm font-bold text-white">Medical Images</h2>
                                                <p className="text-xs text-zinc-500">{uploadedImages.length} image{uploadedImages.length !== 1 ? 's' : ''}</p>
                                            </div>
                                        </div>
                                        <button
                                            onClick={() => setImageSidebarCollapsed(true)}
                                            className="p-2 hover:bg-zinc-800/50 rounded-lg text-zinc-400 hover:text-white transition-all duration-200 hover:scale-105"
                                            title="Collapse sidebar"
                                        >
                                            <ChevronLeft className="h-4 w-4" />
                                        </button>
                                    </div>
                                </div>

                                {/* Image Gallery */}
                                <div className="flex-1 overflow-y-auto custom-scrollbar p-4">
                                    <ImageGallery
                                        images={uploadedImages}
                                        currentIndex={currentImageIndex}
                                        apiBase={API_BASE}
                                        onSelectImage={setCurrentImageIndex}
                                        onUploadClick={() => fileInputRef.current?.click()}
                                    />
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
                            {/* Resize handle */}
                            <div
                                className="absolute right-0 top-0 bottom-0 w-1 hover:w-2 bg-gradient-to-b from-purple-500/20 to-pink-500/20 hover:from-purple-500/50 hover:to-pink-500/50 cursor-col-resize transition-all z-40"
                                onMouseDown={startResize('image')}
                                title="Drag to resize"
                            />
                        </div>
                    )}

                    {/* Expand button for image sidebar when collapsed */}
                    {uploadedImages.length > 0 && imageSidebarCollapsed && (
                        <button
                            onClick={() => setImageSidebarCollapsed(false)}
                            className="fixed top-1/2 -translate-y-1/2 z-10 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white p-3 rounded-r-xl border border-l-0 border-purple-500/50 transition-all duration-200 hover:scale-110 shadow-xl shadow-purple-500/50 hover:shadow-2xl hover:shadow-purple-500/60"
                            style={{
                                left: `${(patientSidebarCollapsed ? 0 : patientSidebarWidth) + (chatSidebarCollapsed ? 0 : chatSidebarWidth)}px`
                            }}
                            title="Show images"
                        >
                            <Image className="h-5 w-5" />
                        </button>
                    )}

                    {/* CENTER: Main Chat Interface (MAIN FOCUS) */}
                    <div className="flex-1 flex flex-col bg-gradient-to-b from-zinc-950 to-zinc-900">
                        {uploadedImages.length === 0 ? (
                            /* Welcome Screen with Upload */
                            <div className="flex items-center justify-center h-full">
                                <div className="text-center max-w-2xl mx-auto p-6">
                                    <div className="relative mb-8">
                                        <div className="absolute inset-0 blur-3xl bg-blue-500/20"></div>
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
                                        className={`relative cursor-pointer border-2 border-dashed rounded-xl p-16 transition-all duration-300 ${dragActive
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
                        ) : (
                            /* Chat Interface - Main Focus */
                            <div className="flex-1 flex flex-col overflow-hidden">
                                {/* Chat Messages Area - WITH SCROLL */}
                                <div className="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar">
                                    {messages.map((msg, idx) => (
                                        <div key={idx} className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''} animate-in slide-in-from-bottom-2 duration-300`}>
                                            {/* Avatar */}
                                            <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center text-white text-sm font-bold shadow-lg ${msg.role === 'user'
                                                ? 'bg-gradient-to-br from-blue-500 to-blue-700'
                                                : msg.role === 'system'
                                                    ? 'bg-gradient-to-br from-zinc-600 to-zinc-800'
                                                    : 'bg-gradient-to-br from-emerald-500 to-emerald-700'
                                                }`}>
                                                {msg.role === 'user' ? '👤' : msg.role === 'system' ? '⚙️' : '🏥'}
                                            </div>

                                            {/* Message Content */}
                                            <div className="flex-1 min-w-0 max-w-3xl">
                                                <div
                                                    className={`inline-block px-5 py-4 rounded-xl text-sm shadow-md ${msg.role === 'user'
                                                        ? 'bg-gradient-to-br from-blue-600 to-blue-700 text-white'
                                                        : msg.role === 'system'
                                                            ? 'bg-zinc-800/80 backdrop-blur-sm text-zinc-300 border border-zinc-700/50'
                                                            : 'bg-zinc-800/80 backdrop-blur-sm text-white border border-zinc-700/50'
                                                        }`}
                                                >
                                                    <MessageRenderer
                                                        content={msg.content}
                                                        apiBase={API_BASE}
                                                        onImageClick={(src, alt) => setModalImage({ src, alt })}
                                                    />
                                                </div>
                                                <div className={`text-xs text-zinc-500 mt-1.5 ${msg.role === 'user' ? 'text-right' : ''}`}>
                                                    {msg.timestamp.toLocaleTimeString()}
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                    <div ref={messagesEndRef} />
                                </div>

                                {/* Chat Input Area */}
                                <div className="border-t border-zinc-800/50 bg-gradient-to-r from-zinc-900/95 to-zinc-900/80 backdrop-blur-sm p-4 space-y-3">
                                    {/* Run Complete Analysis Button - Show when images uploaded but no analysis run */}
                                    {uploadedImages.length > 0 && !isAnalyzing && (
                                        <div className="max-w-4xl mx-auto">
                                            <button
                                                onClick={runCompleteAnalysis}
                                                disabled={isAnalyzing || !currentChatId}
                                                className="w-full px-6 py-4 bg-gradient-to-r from-emerald-600 via-blue-600 to-purple-600 hover:from-emerald-500 hover:via-blue-500 hover:to-purple-500 rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 hover:scale-105 shadow-lg shadow-emerald-500/25 hover:shadow-xl hover:shadow-emerald-500/40 font-semibold flex items-center justify-center gap-2 text-white"
                                            >
                                                <Bot className="h-5 w-5" />
                                                Run Complete AI Analysis
                                                <span className="text-xs opacity-75">({uploadedImages.length} image{uploadedImages.length !== 1 ? 's' : ''})</span>
                                            </button>
                                        </div>
                                    )}

                                    {/* Chat Input */}
                                    <div className="max-w-4xl mx-auto flex gap-3 items-center">
                                        <input
                                            type="text"
                                            value={inputMessage}
                                            onChange={(e) => setInputMessage(e.target.value)}
                                            onKeyPress={(e) => e.key === 'Enter' && !isLoading && sendMessage()}
                                            placeholder="Ask about the analysis..."
                                            className="flex-1 px-5 py-4 bg-zinc-800/50 border border-zinc-700/50 rounded-xl text-sm placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-transparent transition-all backdrop-blur-sm"
                                            disabled={isLoading || !currentChatId}
                                        />
                                        <button
                                            onClick={sendMessage}
                                            disabled={isLoading || !currentChatId || !inputMessage.trim()}
                                            className="px-6 py-4 bg-gradient-to-r from-blue-600 to-emerald-600 hover:from-blue-500 hover:to-emerald-500 rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 hover:scale-105 shadow-lg shadow-blue-500/25 hover:shadow-xl hover:shadow-blue-500/40 font-semibold"
                                        >
                                            {isLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Send className="h-5 w-5" />}
                                        </button>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* RIGHT: Tool Output Panel / Tools Management */}
                    {!toolOutputCollapsed && (
                        <div className="relative flex" style={{ width: `${toolOutputWidth}px` }}>
                            {/* Resize handle (on left side of right panel) */}
                            <div
                                className="absolute left-0 top-0 bottom-0 w-1 hover:w-2 bg-gradient-to-b from-emerald-500/20 to-blue-500/20 hover:from-emerald-500/50 hover:to-blue-500/50 cursor-col-resize transition-all z-40"
                                onMouseDown={startResize('toolOutput')}
                                title="Drag to resize"
                            />
                            <div className="flex-1 transition-all duration-300 ease-in-out border-l border-zinc-800/50 overflow-hidden">
                                <div className="h-full flex flex-col bg-gradient-to-br from-zinc-900/80 to-zinc-900/50 backdrop-blur-sm">
                                    {/* Mode Switcher Header */}
                                    <div className="p-4 border-b border-zinc-800/50 bg-gradient-to-r from-emerald-900/10 via-zinc-900/50 to-blue-900/10">
                                        <div className="flex items-center gap-3 mb-3">
                                            <button
                                                onClick={() => setToolOutputCollapsed(true)}
                                                className="p-2 bg-gradient-to-br from-purple-600/10 to-pink-600/10 border border-purple-500/20 hover:from-purple-600/20 hover:to-pink-600/20 hover:border-purple-500/40 rounded-xl transition-all duration-300 shadow-lg hover:shadow-purple-500/20"
                                                title="Collapse panel"
                                            >
                                                <ChevronRight className="h-4 w-4 text-purple-300" />
                                            </button>
                                            <h2 className="text-sm font-bold text-white flex items-center gap-2 flex-1">
                                                <div className="p-2 rounded-lg bg-gradient-to-br from-emerald-500/10 to-blue-500/10 border border-emerald-500/20">
                                                    <Settings className="h-4 w-4 text-emerald-400" />
                                                </div>
                                                Tools
                                            </h2>
                                        </div>
                                        <div className="flex gap-2 bg-zinc-800/50 rounded-xl p-1.5 backdrop-blur-sm">
                                            <button
                                                onClick={() => setRightSidebarMode('results')}
                                                className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-xl text-xs font-semibold transition-all duration-200 ${rightSidebarMode === 'results'
                                                    ? 'bg-gradient-to-r from-emerald-600 to-blue-600 text-white shadow-lg shadow-emerald-500/25'
                                                    : 'text-zinc-400 hover:text-zinc-300 hover:bg-zinc-700/30'
                                                    }`}
                                            >
                                                Results
                                            </button>
                                            <button
                                                onClick={() => setRightSidebarMode('management')}
                                                className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-xl text-xs font-semibold transition-all duration-200 ${rightSidebarMode === 'management'
                                                    ? 'bg-gradient-to-r from-emerald-600 to-blue-600 text-white shadow-lg shadow-emerald-500/25'
                                                    : 'text-zinc-400 hover:text-zinc-300 hover:bg-zinc-700/30'
                                                    }`}
                                            >
                                                <Settings className="h-3.5 w-3.5" />
                                                Tools
                                            </button>
                                        </div>
                                    </div>

                                    {/* Content Area */}
                                    <div className="flex-1 overflow-hidden">
                                        {rightSidebarMode === 'results' ? (
                                            <ToolOutputPanel
                                                analysisResults={analysisResults}
                                                toolHistory={toolHistory}
                                                filterMode={toolFilterMode}
                                                onFilterModeChange={setToolFilterMode}
                                                currentRequestId={currentRequestId}
                                                currentImage={currentImage}
                                                collapsed={false} // Already controlled by parent
                                                apiBase={API_BASE}
                                                onToggleCollapse={() => { }} // Handled by parent
                                            />
                                        ) : (
                                            <ToolsPanel />
                                        )}
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Expand button for right sidebar when collapsed */}
                    {toolOutputCollapsed && (
                        <button
                            onClick={() => setToolOutputCollapsed(false)}
                            className="fixed right-0 top-1/2 -translate-y-1/2 z-20 bg-gradient-to-l from-blue-600 to-emerald-600 hover:from-blue-500 hover:to-emerald-500 text-white p-3 rounded-l-xl border border-r-0 border-blue-500/50 transition-all duration-200 hover:scale-110 shadow-xl shadow-blue-500/50 hover:shadow-2xl hover:shadow-blue-500/60"
                            title="Show Results"
                        >
                            <ChevronLeft className="h-5 w-5" />
                        </button>
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
