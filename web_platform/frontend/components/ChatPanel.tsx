import { useRef, useEffect } from 'react';
import { MessageSquare, Settings, Trash2, RefreshCw, Loader2 } from 'lucide-react';
import MessageRenderer from './MessageRenderer';
import ToolsPanel from './ToolsPanel';

interface Message {
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: Date;
}

interface ChatPanelProps {
    sessionId: string | null;
    messages: Message[];
    inputMessage: string; // Still needed for parent state
    isLoading: boolean;
    rightSidebarMode: 'chat' | 'tools';
    apiBase: string;
    onInputChange: (value: string) => void; // Still needed for parent state
    onSendMessage: () => void; // Still needed for parent state
    onClearChat: () => void;
    onNewThread: () => void;
    onSetMode: (mode: 'chat' | 'tools') => void;
    onImageClick: (src: string, alt: string) => void;
}

export default function ChatPanel({
    sessionId,
    messages,
    inputMessage,
    isLoading,
    rightSidebarMode,
    apiBase,
    onInputChange,
    onSendMessage,
    onClearChat,
    onNewThread,
    onSetMode,
    onImageClick
}: ChatPanelProps) {
    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    return (
        <div className="w-96 h-full bg-gradient-to-br from-zinc-900/80 to-zinc-900/50 backdrop-blur-sm border-l border-zinc-800/50 flex flex-col">
            {/* Mode Toggle */}
            <div className="p-3 border-b border-zinc-800/50 bg-gradient-to-r from-zinc-900/80 to-zinc-900/60">
                <div className="flex gap-2 bg-zinc-800/50 rounded-xl p-1.5 backdrop-blur-sm">
                    <button
                        onClick={() => onSetMode('chat')}
                        className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold transition-all duration-200 ${rightSidebarMode === 'chat'
                            ? 'bg-gradient-to-r from-blue-600 to-emerald-600 text-white shadow-lg shadow-blue-500/25'
                            : 'text-zinc-400 hover:text-zinc-300 hover:bg-zinc-700/30'
                            }`}
                    >
                        <MessageSquare className="h-4 w-4" />
                        Chat
                    </button>
                    <button
                        onClick={() => onSetMode('tools')}
                        className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold transition-all duration-200 ${rightSidebarMode === 'tools'
                            ? 'bg-gradient-to-r from-blue-600 to-emerald-600 text-white shadow-lg shadow-blue-500/25'
                            : 'text-zinc-400 hover:text-zinc-300 hover:bg-zinc-700/30'
                            }`}
                    >
                        <Settings className="h-4 w-4" />
                        Tools
                    </button>
                </div>
            </div>

            {/* Chat Mode */}
            {rightSidebarMode === 'chat' && (
                <>
                    <div className="p-4 border-b border-zinc-800/50 bg-gradient-to-r from-zinc-900/50 to-zinc-800/30">
                        <div className="flex items-center justify-between">
                            <h2 className="text-sm font-semibold text-zinc-300 flex items-center gap-2">
                                <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
                                Chat with AI
                            </h2>
                            <div className="flex gap-2">
                                <button
                                    onClick={onClearChat}
                                    disabled={messages.length === 0}
                                    className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50 rounded-lg disabled:opacity-30 disabled:cursor-not-allowed transition-all duration-200 hover:scale-105"
                                    title="Clear chat history"
                                >
                                    <Trash2 className="h-3.5 w-3.5" />
                                    Clear
                                </button>
                                <button
                                    onClick={onNewThread}
                                    disabled={!sessionId}
                                    className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50 rounded-lg disabled:opacity-30 disabled:cursor-not-allowed transition-all duration-200 hover:scale-105"
                                    title="Start new conversation thread"
                                >
                                    <RefreshCw className="h-3.5 w-3.5" />
                                    New Thread
                                </button>
                            </div>
                        </div>
                    </div>

                    <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gradient-to-b from-transparent to-zinc-900/30">
                        {messages.map((msg, idx) => (
                            <div key={idx} className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''} animate-in slide-in-from-bottom-2 duration-300`}>
                                {/* Avatar */}
                                <div className={`flex-shrink-0 w-9 h-9 rounded-full flex items-center justify-center text-white text-sm font-bold shadow-lg ${msg.role === 'user'
                                    ? 'bg-gradient-to-br from-blue-500 to-blue-700'
                                    : msg.role === 'system'
                                        ? 'bg-gradient-to-br from-zinc-600 to-zinc-800'
                                        : 'bg-gradient-to-br from-emerald-500 to-emerald-700'
                                    }`}>
                                    {msg.role === 'user' ? '👤' : msg.role === 'system' ? '⚙️' : '🏥'}
                                </div>

                                {/* Message Content */}
                                <div className="flex-1 min-w-0">
                                    <div
                                        className={`inline-block px-4 py-3 rounded-2xl text-sm max-w-full shadow-md ${msg.role === 'user'
                                            ? 'bg-gradient-to-br from-blue-600 to-blue-700 text-white'
                                            : msg.role === 'system'
                                                ? 'bg-zinc-800/80 backdrop-blur-sm text-zinc-300 border border-zinc-700/50'
                                                : 'bg-zinc-800/80 backdrop-blur-sm text-white border border-zinc-700/50'
                                            }`}
                                    >
                                        <MessageRenderer
                                            content={msg.content}
                                            apiBase={apiBase}
                                            onImageClick={(src, alt) => onImageClick(src, alt)}
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

                    {/* Chat Input in sidebar */}
                    <div className="p-4 border-t border-zinc-800/50 bg-gradient-to-r from-zinc-900/80 to-zinc-900/60">
                        <div className="flex gap-3">
                            <input
                                type="text"
                                value={inputMessage}
                                onChange={(e) => onInputChange(e.target.value)}
                                onKeyPress={(e) => e.key === 'Enter' && !isLoading && onSendMessage()}
                                placeholder="Ask about the analysis..."
                                className="flex-1 px-4 py-3 bg-zinc-800/50 border border-zinc-700/50 rounded-xl text-sm placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-transparent transition-all backdrop-blur-sm"
                                disabled={isLoading}
                            />
                            <button
                                onClick={onSendMessage}
                                disabled={isLoading}
                                className="px-5 py-3 bg-gradient-to-r from-blue-600 to-emerald-600 hover:from-blue-500 hover:to-emerald-500 rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 hover:scale-105 shadow-lg shadow-blue-500/25 hover:shadow-xl hover:shadow-blue-500/40"
                            >
                                {isLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : <span className="text-lg">→</span>}
                            </button>
                        </div>
                    </div>
                </>
            )}

            {/* Tools Mode */}
            {rightSidebarMode === 'tools' && (
                <div className="flex-1 flex flex-col overflow-hidden">
                    <ToolsPanel sessionId={sessionId} />
                </div>
            )}
        </div>
    );
}

