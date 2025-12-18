import { useState, useRef } from "react";
import axios from "axios";
import "./Chat.css";

// Helper: 解析產品 Summary 字串為鍵值對
const parseSummary = (summaryText) => {
    if (!summaryText) return [];
    const lines = summaryText.trim().split('\n');
    const details = [];
    lines.forEach(line => {
        // 匹配並解析 Key:Value (支援半形:和全形：)
        const match = line.match(/^(.+?)\s*[:：]\s*(.+)$/); 
        if (match) {
            let key = match[1].trim();
            let value = match[2].trim();
            if (key && value) {
                details.push({ key: key, value: value });
            }
        }
    });
    return details;
};

// 元件: 產品卡片 (Product Card)
const ProductCard = ({ product }) => {
    const details = parseSummary(product.Summary);
    
    // 提取 '商品名稱'
    const title = product.title || details.find(d => d.key === '商品名稱')?.value || '產品資訊缺失';
    
    // 提取 '商品描述'
    const descriptionObject = details.find(d => d.key === '商品描述');
    const description = descriptionObject ? descriptionObject.value : '無詳細描述。';

    return (
        <div className="productCard">
            <div className="productCardTitle">
                {title}
            </div>

            {description && (
                <p className="productCardDescription">
                    {description}
                </p>
            )}

            {product.URL && (
                <a 
                    href={product.URL} 
                    target="_blank" 
                    rel="noopener noreferrer" 
                    className="productCardLink"
                >
                    查看產品詳情 &gt;
                </a>
            )}
        </div>
    );
};


// 元件: 渲染最終總結 (Final Consultation Renderer)
const RenderFinalConsultation = ({ data }) => {
    const { reply, structured_data } = data;
    
    // 後端已合併回覆，故直接使用 reply (不需分割)
    const price = structured_data.predicted_price || 'N/A';
    const recommendations = structured_data.recommendations || [];

    return (
        <div>
            {/* 完整回覆文本 (包含價格宣告和產品引導) */}
            <div className="bot-text-content">
                {reply} 
            </div>

            {/* 價格卡片 */}
            <div className="priceCard">
                預估年保費約為：
                <span className="priceAmount">{price} 元</span>
                <span className="priceDisclaimer">
                    （此為估計值，實際保費可能不同）
                </span>
            </div>

            {/* 產品列表 */}
            {recommendations.length > 0 && (
                <div className="productListContainer">
                    {recommendations.map((p, index) => (
                        <ProductCard key={index} product={p} />
                    ))}
                </div>
            )}
        </div>
    );
};


// 主元件: Chat
export default function Chat() {
    const [messages, setMessages] = useState([
        {
            role: "assistant", 
            type: "chat", 
            text: "您好👋，我是您的 AI 保費預估與產品推薦小幫手！\n\n"+
            "為了給您最精準的建議，請告訴我您的年齡、性別、居住地和預計投保的險種等等(如：壽險、意外險等)。\n\n"+
            "❗️注意：這些資料並不會被我們儲存利用，只會用來預估保費，所以不用擔心，謝謝",
        },
    ]);
    const [input, setInput] = useState("");

    const inputRef = useRef(null); 
    
    // 請在這裡修改 Ngrok 外部網址
    // const BACKEND_URL = "http://localhost:5002"; 
    const BACKEND_URL = "https://heteropolar-dessie-bottlelike.ngrok-free.dev"; 
    
    async function send() {
        if (!input.trim()) return;

        const current = input;
        setInput(""); 
        inputRef.current?.focus(); 

        const userMsg = { role: "user", type: "chat", text: current };
        setMessages((m) => [...m, userMsg]);

        try {
            const res = await axios.post(`${BACKEND_URL}/chat`, {
                message: current,
            });

            const data = res.data;
            
            if (data.complete) {
                setMessages((m) => [
                    ...m,
                    { 
                        role: "assistant", 
                        type: "final_consultation", 
                        data: { reply: data.reply, structured_data: data.structured_data } 
                    },
                ]);
            } else {
                setMessages((m) => [
                    ...m,
                    { role: "assistant", type: "chat", text: data.reply },
                ]);
            }
        } catch (error) {
            console.error("Chat API Error:", error);
            setMessages((m) => [
                ...m,
                { role: "assistant", type: "chat", text: "抱歉，與後端服務連線失敗或發生錯誤。" },
            ]);
        }
    }

    function handleKeyDown(e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            send();
        }
    }

    return (
        <div className="chatWindow">
            <div className="messages">
                {messages.map((m, i) => (
                    <div key={i} className={m.role === "user" ? "bubble user" : "bubble bot"}>
                        {m.type === "final_consultation" ? (
                            // 渲染結構化元件
                            <RenderFinalConsultation data={m.data} />
                        ) : (
                            // 渲染一般文本訊息 (Bot) 或 User 訊息
                            m.role === "bot" || m.role === "assistant" ? (
                                <div className="bot-text-content">{m.text}</div>
                            ) : (
                                <div className="user-text-content">{m.text}</div>
                            )
                        )}
                    </div>
                ))}
            </div>

            <div className="inputBox">
                <textarea
                    ref={inputRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    className="chatInput"
                    rows={1}
                ></textarea>
                
                <button onClick={send}>送出</button>
            </div>
        </div>
    );
}