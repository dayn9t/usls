//! ONNX metadata pre-load reader —— 无需构建 engine 即可读 `metadata_props`。
//!
//! [`Engine::load_onnx`] 返回 `pub(crate)` 的 `onnx::ModelProto`，crate 外不可拼写；
//! 本模块提供一个只返回公有类型的稳定公开函数，供跨 crate 消费者（ml-vision）
//! 在 engine 构建前读取契约（如 `ml.model_contract`）——值须先于 `YOLO::new`
//! 拿到以驱动构建参数（input size），故不能等 session 建好后的 `session.metadata()`。

use std::path::Path;

use anyhow::Result;

use crate::Engine;

/// 从 ONNX 文件读单个 metadata key，不构建 engine。
///
/// 先查 model 级 `metadata_props`，再 fallback 到 graph 级。命中返回值，否则 `None`。
pub fn fetch_metadata_prop<P: AsRef<Path>>(p: P, key: &str) -> Result<Option<String>> {
    let proto = Engine::load_onnx(p)?;
    for entry in &proto.metadata_props {
        if entry.key == key {
            return Ok(Some(entry.value.clone()));
        }
    }
    if let Some(graph) = &proto.graph {
        for entry in &graph.metadata_props {
            if entry.key == key {
                return Ok(Some(entry.value.clone()));
            }
        }
    }
    Ok(None)
}
