"use client"

import { Moon, Sun } from "lucide-react"
import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"

interface SettingsDialogProps {
    open: boolean
    onOpenChange: (open: boolean) => void
    onCloseProtectionChange?: (enabled: boolean) => void
    drawioUi: "min" | "sketch"
    onToggleDrawioUi: () => void
    darkMode: boolean
    onToggleDarkMode: () => void
}

export const STORAGE_ACCESS_CODE_KEY = "next-ai-draw-io-access-code"
export const STORAGE_CLOSE_PROTECTION_KEY = "next-ai-draw-io-close-protection"
const STORAGE_ACCESS_CODE_REQUIRED_KEY = "next-ai-draw-io-access-code-required"
export const STORAGE_AI_PROVIDER_KEY = "next-ai-draw-io-ai-provider"
export const STORAGE_AI_BASE_URL_KEY = "next-ai-draw-io-ai-base-url"
export const STORAGE_AI_API_KEY_KEY = "next-ai-draw-io-ai-api-key"
export const STORAGE_AI_MODEL_KEY = "next-ai-draw-io-ai-model"

function getStoredAccessCodeRequired(): boolean | null {
    if (typeof window === "undefined") return null
    const stored = localStorage.getItem(STORAGE_ACCESS_CODE_REQUIRED_KEY)
    if (stored === null) return null
    return stored === "true"
}

export function SettingsDialog({
    open,
    onOpenChange,
    onCloseProtectionChange,
    drawioUi,
    onToggleDrawioUi,
    darkMode,
    onToggleDarkMode,
}: SettingsDialogProps) {
    const [accessCode, setAccessCode] = useState("")
    const [closeProtection, setCloseProtection] = useState(true)
    const [isVerifying, setIsVerifying] = useState(false)
    const [error, setError] = useState("")
    const [accessCodeRequired, setAccessCodeRequired] = useState(
        () => getStoredAccessCodeRequired() ?? false,
    )
    const [provider, setProvider] = useState("")
    const [baseUrl, setBaseUrl] = useState("")
    const [apiKey, setApiKey] = useState("")
    const [modelId, setModelId] = useState("")

    useEffect(() => {
        // Only fetch if not cached in localStorage
        if (getStoredAccessCodeRequired() !== null) return

        fetch("/api/config")
            .then((res) => {
                if (!res.ok) throw new Error(`HTTP ${res.status}`)
                return res.json()
            })
            .then((data) => {
                const required = data?.accessCodeRequired === true
                localStorage.setItem(
                    STORAGE_ACCESS_CODE_REQUIRED_KEY,
                    String(required),
                )
                setAccessCodeRequired(required)
            })
            .catch(() => {
                // Don't cache on error - allow retry on next mount
                setAccessCodeRequired(false)
            })
    }, [])

    useEffect(() => {
        if (open) {
            const storedCode =
                localStorage.getItem(STORAGE_ACCESS_CODE_KEY) || ""
            setAccessCode(storedCode)

            const storedCloseProtection = localStorage.getItem(
                STORAGE_CLOSE_PROTECTION_KEY,
            )
            // Default to true if not set
            setCloseProtection(storedCloseProtection !== "false")

            // Load AI provider settings
            setProvider(localStorage.getItem(STORAGE_AI_PROVIDER_KEY) || "")
            setBaseUrl(localStorage.getItem(STORAGE_AI_BASE_URL_KEY) || "")
            setApiKey(localStorage.getItem(STORAGE_AI_API_KEY_KEY) || "")
            setModelId(localStorage.getItem(STORAGE_AI_MODEL_KEY) || "")

            setError("")
        }
    }, [open])

    const handleSave = async () => {
        if (!accessCodeRequired) return

        setError("")
        setIsVerifying(true)

        try {
            const response = await fetch("/api/verify-access-code", {
                method: "POST",
                headers: {
                    "x-access-code": accessCode.trim(),
                },
            })

            const data = await response.json()

            if (!data.valid) {
                setError(data.message || "Invalid access code")
                return
            }

            localStorage.setItem(STORAGE_ACCESS_CODE_KEY, accessCode.trim())
            onOpenChange(false)
        } catch {
            setError("Failed to verify access code")
        } finally {
            setIsVerifying(false)
        }
    }

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter") {
            e.preventDefault()
            handleSave()
        }
    }

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="sm:max-w-md">
                <DialogHeader>
                    <DialogTitle>设置</DialogTitle>
                    <DialogDescription>配置你的应用设置。</DialogDescription>
                </DialogHeader>
                <div className="space-y-4 py-2">
                    {accessCodeRequired && (
                        <div className="space-y-2">
                            <Label htmlFor="access-code">访问码</Label>
                            <div className="flex gap-2">
                                <Input
                                    id="access-code"
                                    type="password"
                                    value={accessCode}
                                    onChange={(e) =>
                                        setAccessCode(e.target.value)
                                    }
                                    onKeyDown={handleKeyDown}
                                    placeholder="输入访问码"
                                    autoComplete="off"
                                />
                                <Button
                                    onClick={handleSave}
                                    disabled={isVerifying || !accessCode.trim()}
                                >
                                    {isVerifying ? "..." : "保存"}
                                </Button>
                            </div>
                            <p className="text-[0.8rem] text-muted-foreground">
                                使用本应用需要访问码。
                            </p>
                            {error && (
                                <p className="text-[0.8rem] text-destructive">
                                    {error}
                                </p>
                            )}
                        </div>
                    )}
                    <div className="space-y-2">
                        <Label>AI 提供商设置</Label>
                        <p className="text-[0.8rem] text-muted-foreground">
                            使用你自己的 API
                            密钥来绕过使用限制。你的密钥仅存储在浏览器本地，永远不会存储在服务器上。
                        </p>
                        <div className="space-y-3 pt-2">
                            <div className="space-y-2">
                                <Label htmlFor="ai-provider">提供商</Label>
                                <Select
                                    value={provider || "default"}
                                    onValueChange={(value) => {
                                        const actualValue =
                                            value === "default" ? "" : value
                                        setProvider(actualValue)
                                        localStorage.setItem(
                                            STORAGE_AI_PROVIDER_KEY,
                                            actualValue,
                                        )
                                    }}
                                >
                                    <SelectTrigger id="ai-provider">
                                        <SelectValue placeholder="使用服务器默认" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="default">
                                            使用服务器默认
                                        </SelectItem>
                                        <SelectItem value="openai">
                                            OpenAI
                                        </SelectItem>
                                        <SelectItem value="anthropic">
                                            Anthropic
                                        </SelectItem>
                                        <SelectItem value="google">
                                            Google
                                        </SelectItem>
                                        <SelectItem value="azure">
                                            Azure OpenAI
                                        </SelectItem>
                                        <SelectItem value="openrouter">
                                            OpenRouter
                                        </SelectItem>
                                        <SelectItem value="deepseek">
                                            DeepSeek
                                        </SelectItem>
                                        <SelectItem value="siliconflow">
                                            SiliconFlow
                                        </SelectItem>
                                        <SelectItem value="glm">GLM</SelectItem>
                                        <SelectItem value="qwen">
                                            Qwen
                                        </SelectItem>
                                        <SelectItem value="doubao">
                                            Doubao
                                        </SelectItem>
                                        <SelectItem value="qiniu">
                                            Qiniu (七牛云)
                                        </SelectItem>
                                        <SelectItem value="kimi">
                                            Kimi (月之暗面)
                                        </SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>
                            {provider && provider !== "default" && (
                                <>
                                    <div className="space-y-2">
                                        <Label htmlFor="ai-model">
                                            Model ID
                                        </Label>
                                        <Input
                                            id="ai-model"
                                            value={modelId}
                                            onChange={(e) => {
                                                setModelId(e.target.value)
                                                localStorage.setItem(
                                                    STORAGE_AI_MODEL_KEY,
                                                    e.target.value,
                                                )
                                            }}
                                            placeholder={
                                                provider === "openai"
                                                    ? "e.g., gpt-4o"
                                                    : provider === "anthropic"
                                                      ? "e.g., claude-sonnet-4-5"
                                                      : provider === "google"
                                                        ? "e.g., gemini-2.0-flash-exp"
                                                        : provider ===
                                                            "deepseek"
                                                          ? "e.g., deepseek-chat"
                                                          : provider === "glm"
                                                            ? "e.g., glm-4-plus (or glm-4-0520, glm-4-6)"
                                                            : provider ===
                                                                "qwen"
                                                              ? "e.g., qwen-max (or qwen-turbo, qwen-plus)"
                                                              : provider ===
                                                                  "doubao"
                                                                ? "e.g., doubao-pro-4k (or doubao-pro-32k, doubao-lite-4k)"
                                                                : provider ===
                                                                    "qiniu"
                                                                  ? "e.g., qiniu-deepseek-67b (or qiniu-deepseek-7b, qiniu-deepseek-14b, qiniu-llama-3-70b, qiniu-qwen-72b)"
                                                                  : provider ===
                                                                      "kimi"
                                                                    ? "e.g., moonshot-v1-8k (or moonshot-v1-32k, moonshot-v1-128k)"
                                                                    : "Model ID"
                                            }
                                        />
                                    </div>
                                    <div className="space-y-2">
                                        <Label htmlFor="ai-api-key">
                                            API 密钥
                                        </Label>
                                        <Input
                                            id="ai-api-key"
                                            type="password"
                                            value={apiKey}
                                            onChange={(e) => {
                                                setApiKey(e.target.value)
                                                localStorage.setItem(
                                                    STORAGE_AI_API_KEY_KEY,
                                                    e.target.value,
                                                )
                                            }}
                                            placeholder="你的 API 密钥"
                                            autoComplete="off"
                                        />
                                        <p className="text-[0.8rem] text-muted-foreground">
                                            Overrides{" "}
                                            {provider === "openai"
                                                ? "OPENAI_API_KEY"
                                                : provider === "anthropic"
                                                  ? "ANTHROPIC_API_KEY"
                                                  : provider === "google"
                                                    ? "GOOGLE_GENERATIVE_AI_API_KEY"
                                                    : provider === "azure"
                                                      ? "AZURE_API_KEY"
                                                      : provider ===
                                                          "openrouter"
                                                        ? "OPENROUTER_API_KEY"
                                                        : provider ===
                                                            "deepseek"
                                                          ? "DEEPSEEK_API_KEY"
                                                          : provider ===
                                                              "siliconflow"
                                                            ? "SILICONFLOW_API_KEY"
                                                            : provider === "glm"
                                                              ? "GLM_API_KEY"
                                                              : provider ===
                                                                  "qwen"
                                                                ? "QWEN_API_KEY"
                                                                : provider ===
                                                                    "doubao"
                                                                  ? "DOUBAO_API_KEY"
                                                                  : provider ===
                                                                      "qiniu"
                                                                    ? "QINIU_API_KEY"
                                                                    : provider ===
                                                                        "kimi"
                                                                      ? "KIMI_API_KEY"
                                                                      : "server API key"}
                                        </p>
                                    </div>
                                    <div className="space-y-2">
                                        <Label htmlFor="ai-base-url">
                                            Base URL (可选)
                                        </Label>
                                        <Input
                                            id="ai-base-url"
                                            value={baseUrl}
                                            onChange={(e) => {
                                                setBaseUrl(e.target.value)
                                                localStorage.setItem(
                                                    STORAGE_AI_BASE_URL_KEY,
                                                    e.target.value,
                                                )
                                            }}
                                            placeholder={
                                                provider === "anthropic"
                                                    ? "https://api.anthropic.com/v1"
                                                    : provider === "siliconflow"
                                                      ? "https://api.siliconflow.com/v1"
                                                      : provider === "glm"
                                                        ? "https://open.bigmodel.cn/api/paas/v4"
                                                        : provider === "qwen"
                                                          ? "https://dashscope.aliyun.com/compatible-mode/v1"
                                                          : provider ===
                                                              "doubao"
                                                            ? "https://ark.cn-beijing.volces.com/api/v3"
                                                            : provider ===
                                                                "qiniu"
                                                              ? "https://api.qiniucdn.com/v1"
                                                              : provider ===
                                                                  "kimi"
                                                                ? "https://api.moonshot.cn/v1"
                                                                : "Custom endpoint URL"
                                            }
                                        />
                                    </div>
                                    <Button
                                        variant="outline"
                                        size="sm"
                                        className="w-full"
                                        onClick={() => {
                                            localStorage.removeItem(
                                                STORAGE_AI_PROVIDER_KEY,
                                            )
                                            localStorage.removeItem(
                                                STORAGE_AI_BASE_URL_KEY,
                                            )
                                            localStorage.removeItem(
                                                STORAGE_AI_API_KEY_KEY,
                                            )
                                            localStorage.removeItem(
                                                STORAGE_AI_MODEL_KEY,
                                            )
                                            setProvider("")
                                            setBaseUrl("")
                                            setApiKey("")
                                            setModelId("")
                                        }}
                                    >
                                        清除设置
                                    </Button>
                                </>
                            )}
                        </div>
                    </div>

                    <div className="flex items-center justify-between">
                        <div className="space-y-0.5">
                            <Label htmlFor="theme-toggle">主题</Label>
                            <p className="text-[0.8rem] text-muted-foreground">
                                界面和 DrawIO 画布的深色/浅色模式。
                            </p>
                        </div>
                        <Button
                            id="theme-toggle"
                            variant="outline"
                            size="icon"
                            onClick={onToggleDarkMode}
                        >
                            {darkMode ? (
                                <Sun className="h-4 w-4" />
                            ) : (
                                <Moon className="h-4 w-4" />
                            )}
                        </Button>
                    </div>

                    <div className="flex items-center justify-between">
                        <div className="space-y-0.5">
                            <Label htmlFor="drawio-ui">DrawIO 样式</Label>
                            <p className="text-[0.8rem] text-muted-foreground">
                                画布样式: {drawioUi === "min" ? "简约" : "素描"}
                            </p>
                        </div>
                        <Button
                            id="drawio-ui"
                            variant="outline"
                            size="sm"
                            onClick={onToggleDrawioUi}
                        >
                            切换到 {drawioUi === "min" ? "素描" : "简约"}
                        </Button>
                    </div>

                    <div className="flex items-center justify-between">
                        <div className="space-y-0.5">
                            <Label htmlFor="close-protection">关闭保护</Label>
                            <p className="text-[0.8rem] text-muted-foreground">
                                离开页面时显示确认提示。
                            </p>
                        </div>
                        <Switch
                            id="close-protection"
                            checked={closeProtection}
                            onCheckedChange={(checked) => {
                                setCloseProtection(checked)
                                localStorage.setItem(
                                    STORAGE_CLOSE_PROTECTION_KEY,
                                    checked.toString(),
                                )
                                onCloseProtectionChange?.(checked)
                            }}
                        />
                    </div>
                </div>
            </DialogContent>
        </Dialog>
    )
}
