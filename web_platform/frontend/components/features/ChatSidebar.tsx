'use client';

import { useState } from 'react';
import { MessageSquare, Plus, Trash2, ChevronLeft, Image as ImageIcon, Clock } from 'lucide-react';

interface Chat {
    chat_id: string;
    name: string;
    created_at: string;
    last_access: string | null;
    message_count: number;
    image_count: number;
}

interface ChatSidebarProps {
    userId: string;
    currentChatId: string | null;
    chats: Chat[];
    collapsed: boolean;
    onSelectChat: (chatId: string) => void;
    onNewChat: () => void;
    onDeleteChat: (chatId: string) => void;
    onToggleCollapse: () => void;
}

export default function ChatSidebar({
    userId,
    currentChatId,
    chats,
    collapsed,
    onSelectChat,
    onNewChat,
    onDeleteChat,
    onToggleCollapse
}: ChatSidebarProps) {
    const [hoveredChat, setHoveredChat] = useState<string | null>(null);

    return (
        <div className={`transition-all duration-300 ease-in-out bg-gradient-to-b from-zinc-900 via-zinc-900/95 to-zinc-900/90 border-r border-zinc-800/50 flex flex-col shadow-xl ${collapsed ? 'w-0' : 'w-80'} overflow-hidden`}>
            {/* Header with gradient accent */}
            <div className="relative p-4 border-b border-zinc-800/50 bg-gradient-to-r from-emerald-900/20 via-zinc-900/50 to-blue-900/20">
                <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                        <div className="p-2 rounded-lg bg-gradient-to-br from-emerald-500/10 to-blue-500/10 border border-emerald-500/20">
                            <MessageSquare className="h-4 w-4 text-emerald-400" />
                        </div>
                        <div>
                            <h2 className="text-sm font-bold text-white">Conversations</h2>
                            <p className="text-xs text-zinc-500">{chats.length} chat{chats.length !== 1 ? 's' : ''}</p>
                        </div>
                    </div>
                    <button
                        onClick={onToggleCollapse}
                        className="p-2 hover:bg-zinc-800/50 rounded-lg text-zinc-400 hover:text-white transition-all duration-200 hover:scale-105"
                        title="Collapse sidebar"
                    >
                        <ChevronLeft className="h-4 w-4" />
                    </button>
                </div>

                <button
                    onClick={onNewChat}
                    className="w-full px-4 py-3 bg-gradient-to-r from-emerald-600 to-blue-600 hover:from-emerald-500 hover:to-blue-500 rounded-xl text-sm font-semibold flex items-center justify-center gap-2 transition-all duration-200 hover:scale-105 shadow-lg shadow-emerald-500/25 hover:shadow-xl hover:shadow-emerald-500/40"
                >
                    <Plus className="h-4 w-4" />
                    New Conversation
                </button>
            </div>

            {/* Chat List with modern styling */}
            <div className="flex-1 overflow-y-auto custom-scrollbar p-2 space-y-1">
                {chats.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full p-6 text-center">
                        <div className="p-4 rounded-full bg-zinc-800/50 mb-3">
                            <MessageSquare className="h-8 w-8 text-zinc-600" />
                        </div>
                        <p className="text-sm text-zinc-500">No conversations yet</p>
                        <p className="text-xs text-zinc-600 mt-1">Create a new chat above!</p>
                    </div>
                ) : (
                    chats.map((chat) => (
                        <div
                            key={chat.chat_id}
                            onClick={() => onSelectChat(chat.chat_id)}
                            onMouseEnter={() => setHoveredChat(chat.chat_id)}
                            onMouseLeave={() => setHoveredChat(null)}
                            className={`group relative p-3 rounded-xl cursor-pointer transition-all duration-200 ${currentChatId === chat.chat_id
                                ? 'bg-gradient-to-r from-emerald-600/20 via-emerald-500/10 to-blue-600/20 border border-emerald-500/30 shadow-lg shadow-emerald-500/10'
                                : 'hover:bg-zinc-800/50 border border-transparent hover:border-zinc-700/50'
                                }`}
                        >
                            {/* Active indicator */}
                            {currentChatId === chat.chat_id && (
                                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-12 bg-gradient-to-b from-emerald-500 to-blue-500 rounded-r" />
                            )}

                            <div className="flex items-start gap-3">
                                {/* Chat Icon */}
                                <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${currentChatId === chat.chat_id
                                    ? 'bg-gradient-to-br from-emerald-500 to-blue-500 text-white'
                                    : 'bg-zinc-800 text-zinc-400 group-hover:bg-zinc-700'
                                    }`}>
                                    <MessageSquare className="h-4 w-4" />
                                </div>

                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center justify-between gap-2 mb-1">
                                        <h3 className={`font-semibold text-sm truncate ${currentChatId === chat.chat_id ? 'text-white' : 'text-zinc-300'
                                            }`}>
                                            {chat.name}
                                        </h3>

                                        {/* Delete button */}
                                        {hoveredChat === chat.chat_id && currentChatId !== chat.chat_id && (
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    onDeleteChat(chat.chat_id);
                                                }}
                                                className="p-1.5 hover:bg-red-500/20 rounded-lg transition-colors flex-shrink-0"
                                                title="Delete chat"
                                            >
                                                <Trash2 className="h-3.5 w-3.5 text-red-400" />
                                            </button>
                                        )}
                                    </div>

                                    {/* Metadata badges */}
                                    <div className="flex items-center gap-2 mb-2">
                                        {chat.message_count > 0 && (
                                            <span className="px-2 py-0.5 rounded-full bg-zinc-800/50 text-xs text-zinc-400 flex items-center gap-1">
                                                <MessageSquare className="h-3 w-3" />
                                                {chat.message_count}
                                            </span>
                                        )}
                                        {chat.image_count > 0 && (
                                            <span className="px-2 py-0.5 rounded-full bg-blue-500/10 text-xs text-blue-400 flex items-center gap-1 border border-blue-500/20">
                                                <ImageIcon className="h-3 w-3" />
                                                {chat.image_count}
                                            </span>
                                        )}
                                    </div>

                                    {/* Timestamp */}
                                    {chat.last_access && (
                                        <div className="text-xs text-zinc-600 flex items-center gap-1">
                                            <Clock className="h-3 w-3" />
                                            {new Date(chat.last_access).toLocaleString('en-US', {
                                                month: 'short',
                                                day: 'numeric',
                                                hour: '2-digit',
                                                minute: '2-digit'
                                            })}
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}

