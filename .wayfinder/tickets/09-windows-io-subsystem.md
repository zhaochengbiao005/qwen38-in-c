# Ticket-09: Windows I/O 子系统原型（task, AFK）

**Status: wontfix（全驻留设计裁剪）**

Ticket-04 定案全驻留设计（FP8+scale 常驻 ~25GB，内存地板 32GB），流式权重管线为后置增强不进一阶段验收。引擎实测 RSS 25.3GB，加载 3.4s（mmap + 按需调页），I/O 不是瓶颈。若日后要降到 16GB 以下或做 serving，再立票。
