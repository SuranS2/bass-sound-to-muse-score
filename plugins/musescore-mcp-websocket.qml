/*
 * MuseScore 4 MCP WebSocket Plugin (Loopback-Only)
 *
 * Adapted from ghchen99/mcp-musescore (https://github.com/ghchen99/mcp-musescore)
 * Copyright (c) 2025 ghchen99
 * Licensed under the MIT License.
 *
 * Security modification: Explicitly binds only to 127.0.0.1 (loopback) on port 8765.
 * Install into: %USERPROFILE%\Documents\MuseScore4\Plugins\musescore-mcp-websocket.qml
 */
import QtQuick 2.9
import QtWebSockets 1.1
import MuseScore 3.0

MuseScore {
    menuPath: "Plugins.Bass MCP WebSocket Server"
    description: "Loopback WebSocket server (127.0.0.1:8765) for bass-sound-to-muse-score MCP control."
    version: "1.0"
    requiresScore: false

    WebSocketServer {
        id: server
        host: "127.0.0.1"
        port: 8765
        listen: true
        accept: true

        onClientConnected: function(webSocket) {
            webSocket.onTextMessageReceived.connect(function(message) {
                var response = handleCommand(message);
                webSocket.sendTextMessage(JSON.stringify(response));
            });
        }
    }

    function handleCommand(rawMessage) {
        var req;
        try {
            req = JSON.parse(rawMessage);
        } catch (e) {
            return { status: "error", message: "Invalid JSON payload" };
        }

        var action = req.action || "";
        var params = req.params || {};

        if (action === "ping") {
            return { status: "ok", message: "pong", hasScore: !!curScore };
        }
        if (!curScore) {
            return { status: "error", message: "No active score open in MuseScore" };
        }

        if (action === "get_score") {
            return {
                status: "ok",
                title: curScore.title || "",
                measures: curScore.nmeasures || 0,
                staves: curScore.nstaves || 0
            };
        }

        curScore.startCmd();
        try {
            if (action === "set_title") {
                curScore.title = String(params.title || "Bass Transcription");
            } else if (action === "append_measure") {
                curScore.appendMeasures(Number(params.count || 1));
            } else if (action === "add_note") {
                var cursor = curScore.newCursor();
                cursor.rewind(0);
                cursor.setDuration(1, 4);
                cursor.addNote(Number(params.pitch || 40));
            } else if (action === "add_rest") {
                var restCursor = curScore.newCursor();
                restCursor.rewind(0);
                restCursor.setDuration(1, 4);
                restCursor.addRest();
            } else {
                curScore.endCmd(true);
                return { status: "error", message: "Unsupported action: " + action };
            }
            curScore.endCmd();
            return { status: "ok", action: action };
        } catch (err) {
            curScore.endCmd(true);
            return { status: "error", message: String(err) };
        }
    }

    onRun: {
        console.log("Bass MCP WebSocket server listening on ws://127.0.0.1:8765");
    }
}
