'use strict';
/* Small standalone catalog adapted from HiveMind's MIT-licensed locale behavior.
 * Titanos keeps only its own user journey; no React/i18next runtime is copied. */
globalThis.TitanosI18n=(()=>{
const locales=['en','es','ja','zh-CN'],labels={en:'English',es:'Español',ja:'日本語','zh-CN':'简体中文'};
const en={
'brand.home':'Titanos home','brand.runtime':'Standalone local inference','connection.connecting':'Connecting…','connection.connected':'Connected, {{count}} models','connection.needed':'Connection needed','action.refresh':'Refresh',
'mode.label':'Mode','mode.simple':'Simple','mode.advanced':'Advanced','language.label':'Language','models.title':'Models','models.empty':'No models yet. Use Add a model, or run titanos pull.','models.add':'Register a local path','connection.title':'Connection','connection.key':'Server API key','connection.connect':'Connect','connection.memory':'Kept in this tab only.',
'nav.api':'API reference','nav.openapi':'OpenAPI schema','models.technical':'Details','setup.review':'Review','setup.download':'Download and run','setup.cancel':'Cancel download',
'workspace.title':'Run a model','setup.title':'Add a model','setup.intro':'Pick a task and a source, then search. Size and license are shown before anything is downloaded.','setup.task':'Task','setup.source':'Source','setup.search':'Search terms','setup.find':'Search','setup.link':'Model page link','setup.inspect':'Inspect','setup.ready':'Ready. Select it under Model.','setup.retry':'Retry',
'task.chat':'Chat and write','task.vision':'Understand an image','task.image':'Make or edit an image','task.video':'Make a video','task.transcription':'Transcribe audio','task.speech':'Speak text','task.embeddings':'Create embeddings','task.translation':'Translate text','task.image-to-text':'Read or caption an image','task.object-detection':'Detect objects','task.depth-estimation':'Estimate relative depth','task.image-segmentation':'Create an edit mask','task.sentiment':'Classify sentiment','task.rerank':'Rerank passages','task.pair-classification':'Compare two texts','task.vad':'Detect speech activity','task.music':'Create music or a song',
'composer.model':'Model','composer.prompt':'Prompt','composer.placeholder':'Type a prompt','composer.settings':'Settings','composer.generate':'Run','composer.stop':'Stop','composer.clear':'Clear','composer.unload':'Unload','composer.load':'Load','composer.details':'Runtime','composer.remove':'Remove',
'jobs.title':'Jobs','jobs.refresh':'Refresh','jobs.empty':'No jobs yet. Image, audio, music and video runs are listed here while they run and after they finish.','jobs.open':'Open','jobs.cancel':'Cancel','status.ready':'Ready','status.queued':'Queued','status.running':'Running…','status.cancelling':'Cancelling…','status.completed':'Completed','status.failed':'Failed','status.interrupted':'Interrupted','status.done':'Done','status.error':'Request failed','status.cancelled':'Cancelled','status.connected':'Connected. Select a model.',
'advanced.title':'Advanced configuration','advanced.devices':'Device assignment','advanced.mode':'Execution mode','advanced.single':'One model on one device','advanced.sharding':'Split one model across devices','advanced.tensor':'Tensor parallel ranks','advanced.replicas':'Independent replicas','advanced.apply':'Save assignment','advanced.loadDevices':'Load available devices','advanced.auto':'Automatic','advanced.missing':'A saved device is unavailable. Choose an available device or restore it before loading.','advanced.identity':'Stable device identity','advanced.resources':'Resources and diagnostics','firstRun.title':'First run','firstRun.copy':'Choose a compatible model above or register a complete local bundle. Titanos never downloads during inference, and removing a registration does not delete model files.',
'music.mode':'Music type','music.instrumental':'Instrumental','music.song':'Song with vocals','music.lyrics':'Lyrics (optional when supported)','music.duration':'Duration in seconds','music.language':'Vocal language','music.tempo':'Tempo (BPM)','music.key':'Key / scale','music.reference':'Reference audio','music.create':'Create music',
'a11y.selectModel':'Select {{model}}, {{task}}, {{status}}','a11y.openJob':'Open {{task}} job {{id}} for {{model}}','a11y.cancelJob':'Cancel {{task}} job {{id}} for {{model}}','footer.independent':'Titanos runs on its own. It does not need PAIR or HiveMind.'
,'error.model_not_loaded':'Load this model first or allow lazy loading.','error.registry_busy':'Finish or cancel active work, then retry.','error.registry_read_only':'Restart Titanos with a writable model registry.','error.model_not_found':'Refresh the catalog or register the checkpoint again.','error.unsupported_capability':'Choose a compatible model or use the documented API/CLI path.','error.request_too_large':'Reduce the input or control value to the active limit.','error.validation':'Correct the named fields and active limits.','error.unavailable':'Check the complete model, runtime and selected device.'
};
const es={
'brand.home':'Inicio de Titanos','brand.runtime':'Inferencia local independiente','connection.connecting':'Conectando…','connection.connected':'Conectado, {{count}} modelos','connection.needed':'Se necesita conexión','action.refresh':'Actualizar',
'mode.label':'Modo','mode.simple':'Simple','mode.advanced':'Avanzado','language.label':'Idioma','models.title':'Modelos','models.empty':'Aún no hay modelos. Usa Añadir un modelo o ejecuta titanos pull.','models.add':'Registrar una ruta local','connection.title':'Conexión','connection.key':'Clave API del servidor','connection.connect':'Conectar','connection.memory':'Solo se conserva en esta pestaña.',
'nav.api':'Referencia de API','nav.openapi':'Esquema OpenAPI','models.technical':'Detalles','setup.review':'Revisar','setup.download':'Descargar y ejecutar','setup.cancel':'Cancelar descarga',
'workspace.title':'Ejecutar un modelo','setup.title':'Añadir un modelo','setup.intro':'Elige una tarea y una fuente, luego busca. El tamaño y la licencia se muestran antes de descargar nada.','setup.task':'Tarea','setup.source':'Fuente','setup.search':'Términos de búsqueda','setup.find':'Buscar','setup.link':'Enlace a la página del modelo','setup.inspect':'Inspeccionar','setup.ready':'Listo. Selecciónalo en Modelo.','setup.retry':'Reintentar',
'task.chat':'Chatear y escribir','task.vision':'Comprender una imagen','task.image':'Crear o editar una imagen','task.video':'Crear un vídeo','task.transcription':'Transcribir audio','task.speech':'Convertir texto en voz','task.embeddings':'Crear representaciones vectoriales','task.translation':'Traducir texto','task.image-to-text':'Leer o describir una imagen','task.object-detection':'Detectar objetos','task.depth-estimation':'Estimar profundidad relativa','task.image-segmentation':'Crear una máscara de edición','task.sentiment':'Clasificar sentimiento','task.rerank':'Reordenar pasajes','task.pair-classification':'Comparar dos textos','task.vad':'Detectar actividad de voz','task.music':'Crear música o una canción',
'composer.model':'Modelo','composer.prompt':'Indicación','composer.placeholder':'Escribe una indicación','composer.settings':'Ajustes','composer.generate':'Ejecutar','composer.stop':'Detener','composer.clear':'Limpiar','composer.unload':'Descargar','composer.load':'Cargar','composer.details':'Entorno de ejecución','composer.remove':'Eliminar',
'jobs.title':'Trabajos','jobs.refresh':'Actualizar','jobs.empty':'Aún no hay trabajos. Las ejecuciones de imagen, audio, música y vídeo se listan aquí mientras se ejecutan y al terminar.','jobs.open':'Abrir','jobs.cancel':'Cancelar','status.ready':'Listo','status.queued':'En cola','status.running':'En ejecución…','status.cancelling':'Cancelando…','status.completed':'Completado','status.failed':'Fallido','status.interrupted':'Interrumpido','status.done':'Terminado','status.error':'La solicitud falló','status.cancelled':'Cancelado','status.connected':'Conectado. Selecciona un modelo.',
'advanced.title':'Configuración avanzada','advanced.devices':'Asignación de dispositivos','advanced.mode':'Modo de ejecución','advanced.single':'Un modelo en un dispositivo','advanced.sharding':'Dividir un modelo entre dispositivos','advanced.tensor':'Rangos de paralelismo tensorial','advanced.replicas':'Réplicas independientes','advanced.apply':'Guardar asignación','advanced.loadDevices':'Cargar dispositivos disponibles','advanced.auto':'Automático','advanced.missing':'Un dispositivo guardado no está disponible. Elige uno disponible o restáuralo antes de cargar.','advanced.identity':'Identidad estable del dispositivo','advanced.resources':'Recursos y diagnósticos','firstRun.title':'Primer uso','firstRun.copy':'Elige arriba un modelo compatible o registra un paquete local completo. Titanos no descarga durante la inferencia y eliminar un registro no borra los archivos del modelo.',
'music.mode':'Tipo de música','music.instrumental':'Instrumental','music.song':'Canción con voz','music.lyrics':'Letra (opcional cuando sea compatible)','music.duration':'Duración en segundos','music.language':'Idioma de la voz','music.tempo':'Tempo (BPM)','music.key':'Tonalidad / escala','music.reference':'Audio de referencia','music.create':'Crear música',
'a11y.selectModel':'Seleccionar {{model}}, {{task}}, {{status}}','a11y.openJob':'Abrir trabajo {{task}} {{id}} de {{model}}','a11y.cancelJob':'Cancelar trabajo {{task}} {{id}} de {{model}}','footer.independent':'Titanos funciona por sí solo. No necesita PAIR ni HiveMind.'
,'error.model_not_loaded':'Carga primero este modelo o permite la carga automática.','error.registry_busy':'Finaliza o cancela el trabajo activo y vuelve a intentarlo.','error.registry_read_only':'Reinicia Titanos con un registro de modelos editable.','error.model_not_found':'Actualiza el catálogo o vuelve a registrar el modelo.','error.unsupported_capability':'Elige un modelo compatible o usa la ruta API/CLI documentada.','error.request_too_large':'Reduce la entrada o el control al límite activo.','error.validation':'Corrige los campos indicados y los límites activos.','error.unavailable':'Comprueba el modelo completo, el entorno y el dispositivo elegido.'
};
const ja={
'brand.home':'Titanos ホーム','brand.runtime':'スタンドアロン ローカル推論','connection.connecting':'接続中…','connection.connected':'接続済み、{{count}} モデル','connection.needed':'接続が必要です','action.refresh':'更新',
'mode.label':'モード','mode.simple':'シンプル','mode.advanced':'詳細','language.label':'言語','models.title':'モデル','models.empty':'モデルはまだありません。「モデルを追加」を使うか、titanos pull を実行してください。','models.add':'ローカルパスを登録','connection.title':'接続','connection.key':'サーバー API キー','connection.connect':'接続','connection.memory':'このタブだけに保持します。',
'nav.api':'API リファレンス','nav.openapi':'OpenAPI スキーマ','models.technical':'詳細','setup.review':'内容を確認','setup.download':'ダウンロードして実行','setup.cancel':'ダウンロードをキャンセル',
'workspace.title':'モデルを実行','setup.title':'モデルを追加','setup.intro':'タスクと提供元を選んで検索します。ダウンロード前に容量とライセンスを表示します。','setup.task':'タスク','setup.source':'提供元','setup.search':'検索語','setup.find':'検索','setup.link':'モデルページのリンク','setup.inspect':'リンクを検査','setup.ready':'準備完了。「モデル」で選択してください。','setup.retry':'再試行',
'task.chat':'会話と文章作成','task.vision':'画像を理解','task.image':'画像を作成・編集','task.video':'動画を作成','task.transcription':'音声を書き起こす','task.speech':'文章を音声にする','task.embeddings':'埋め込みを作成','task.translation':'文章を翻訳','task.image-to-text':'画像を読む・説明する','task.object-detection':'物体を検出','task.depth-estimation':'相対深度を推定','task.image-segmentation':'編集マスクを作成','task.sentiment':'感情を分類','task.rerank':'文章を再順位付け','task.pair-classification':'二つの文章を比較','task.vad':'音声区間を検出','task.music':'音楽や歌を作成',
'composer.model':'モデル','composer.prompt':'指示','composer.placeholder':'指示を入力','composer.settings':'設定','composer.generate':'実行','composer.stop':'停止','composer.clear':'クリア','composer.unload':'アンロード','composer.load':'ロード','composer.details':'実行環境','composer.remove':'削除',
'jobs.title':'ジョブ','jobs.refresh':'更新','jobs.empty':'ジョブはまだありません。画像、音声、音楽、動画の実行が、実行中と完了後にここへ表示されます。','jobs.open':'開く','jobs.cancel':'キャンセル','status.ready':'準備完了','status.queued':'待機中','status.running':'実行中…','status.cancelling':'キャンセル中…','status.completed':'完了','status.failed':'失敗','status.interrupted':'中断','status.done':'完了','status.error':'リクエストに失敗しました','status.cancelled':'キャンセル済み','status.connected':'接続しました。モデルを選んでください。',
'advanced.title':'詳細設定','advanced.devices':'デバイス割り当て','advanced.mode':'実行モード','advanced.single':'一つのデバイスで一つのモデル','advanced.sharding':'一つのモデルを複数デバイスに分割','advanced.tensor':'テンソル並列ランク','advanced.replicas':'独立レプリカ','advanced.apply':'割り当てを保存','advanced.loadDevices':'利用可能なデバイスを読み込む','advanced.auto':'自動','advanced.missing':'保存済みデバイスがありません。利用可能なデバイスを選ぶか、復旧してからロードしてください。','advanced.identity':'安定したデバイス ID','advanced.resources':'リソースと診断','firstRun.title':'初回利用','firstRun.copy':'上で互換モデルを選ぶか、完全なローカルバンドルを登録してください。推論中にダウンロードせず、登録を削除してもモデルファイルは消しません。',
'music.mode':'音楽の種類','music.instrumental':'インストゥルメンタル','music.song':'ボーカル曲','music.lyrics':'歌詞（対応時は任意）','music.duration':'長さ（秒）','music.language':'歌唱言語','music.tempo':'テンポ（BPM）','music.key':'キー / スケール','music.reference':'参照音声','music.create':'音楽を作成',
'a11y.selectModel':'{{model}}、{{task}}、{{status}} を選択','a11y.openJob':'{{model}} の {{task}} ジョブ {{id}} を開く','a11y.cancelJob':'{{model}} の {{task}} ジョブ {{id}} をキャンセル','footer.independent':'Titanos は単独で動作します。PAIR や HiveMind は不要です。'
,'error.model_not_loaded':'先にモデルをロードするか、自動ロードを許可してください。','error.registry_busy':'実行中の処理を完了またはキャンセルして再試行してください。','error.registry_read_only':'書き込み可能なモデル登録で Titanos を再起動してください。','error.model_not_found':'カタログを更新するか、モデルを再登録してください。','error.unsupported_capability':'互換モデルを選ぶか、文書化された API/CLI を使用してください。','error.request_too_large':'入力または設定値を現在の上限まで減らしてください。','error.validation':'指定された項目と現在の上限を修正してください。','error.unavailable':'完全なモデル、実行環境、選択デバイスを確認してください。'
};
const zh={
'brand.home':'Titanos 主页','brand.runtime':'独立本地推理','connection.connecting':'正在连接…','connection.connected':'已连接，{{count}} 个模型','connection.needed':'需要连接','action.refresh':'刷新',
'mode.label':'模式','mode.simple':'简单','mode.advanced':'高级','language.label':'语言','models.title':'模型','models.empty':'还没有模型。请使用“添加模型”，或运行 titanos pull。','models.add':'注册本地路径','connection.title':'连接','connection.key':'服务器 API 密钥','connection.connect':'连接','connection.memory':'仅保存在此标签页。',
'nav.api':'API 参考','nav.openapi':'OpenAPI 架构','models.technical':'详情','setup.review':'查看','setup.download':'下载并运行','setup.cancel':'取消下载',
'workspace.title':'运行模型','setup.title':'添加模型','setup.intro':'选择任务和来源，然后搜索。下载前会显示大小和许可信息。','setup.task':'任务','setup.source':'来源','setup.search':'搜索词','setup.find':'搜索','setup.link':'模型页面链接','setup.inspect':'检查','setup.ready':'已就绪。请在“模型”中选择它。','setup.retry':'重试',
'task.chat':'对话与写作','task.vision':'理解图像','task.image':'创建或编辑图像','task.video':'创建视频','task.transcription':'转录音频','task.speech':'文字转语音','task.embeddings':'创建嵌入','task.translation':'翻译文本','task.image-to-text':'读取或描述图像','task.object-detection':'检测对象','task.depth-estimation':'估计相对深度','task.image-segmentation':'创建编辑蒙版','task.sentiment':'情感分类','task.rerank':'段落重排','task.pair-classification':'比较两段文本','task.vad':'检测语音活动','task.music':'创作音乐或歌曲',
'composer.model':'模型','composer.prompt':'提示','composer.placeholder':'输入提示','composer.settings':'设置','composer.generate':'运行','composer.stop':'停止','composer.clear':'清除','composer.unload':'卸载','composer.load':'加载','composer.details':'运行时','composer.remove':'删除',
'jobs.title':'任务','jobs.refresh':'刷新','jobs.empty':'还没有任务。图像、音频、音乐和视频的运行会在运行中和完成后列在这里。','jobs.open':'打开','jobs.cancel':'取消','status.ready':'已准备好','status.queued':'排队中','status.running':'正在运行…','status.cancelling':'正在取消…','status.completed':'已完成','status.failed':'失败','status.interrupted':'已中断','status.done':'完成','status.error':'请求失败','status.cancelled':'已取消','status.connected':'已连接。请选择模型。',
'advanced.title':'高级配置','advanced.devices':'设备分配','advanced.mode':'执行模式','advanced.single':'一个设备运行一个模型','advanced.sharding':'将一个模型拆分到多个设备','advanced.tensor':'张量并行秩','advanced.replicas':'独立副本','advanced.apply':'保存分配','advanced.loadDevices':'加载可用设备','advanced.auto':'自动','advanced.missing':'已保存的设备不可用。请先选择可用设备或恢复该设备。','advanced.identity':'稳定设备标识','advanced.resources':'资源与诊断','firstRun.title':'首次使用','firstRun.copy':'请在上方选择兼容模型，或注册完整的本地组件包。Titanos 不会在推理期间下载内容，删除注册也不会删除模型文件。',
'music.mode':'音乐类型','music.instrumental':'纯音乐','music.song':'人声歌曲','music.lyrics':'歌词（支持时可选）','music.duration':'时长（秒）','music.language':'演唱语言','music.tempo':'速度（BPM）','music.key':'调 / 音阶','music.reference':'参考音频','music.create':'创作音乐',
'a11y.selectModel':'选择 {{model}}、{{task}}、{{status}}','a11y.openJob':'打开 {{model}} 的 {{task}} 任务 {{id}}','a11y.cancelJob':'取消 {{model}} 的 {{task}} 任务 {{id}}','footer.independent':'Titanos 可独立运行，无需 PAIR 或 HiveMind。'
,'error.model_not_loaded':'请先加载此模型，或允许自动加载。','error.registry_busy':'请先完成或取消正在运行的任务，然后重试。','error.registry_read_only':'请使用可写模型注册表重启 Titanos。','error.model_not_found':'请刷新目录或重新注册模型。','error.unsupported_capability':'请选择兼容模型，或使用文档中的 API/CLI。','error.request_too_large':'请将输入或控制值减小到当前限制内。','error.validation':'请修正指定字段并遵守当前限制。','error.unavailable':'请检查完整模型、运行时和所选设备。'
};
en['advanced.unsaved']='Unsaved assignment';es['advanced.unsaved']='Asignación sin guardar';
ja['advanced.unsaved']='未保存の割り当て';zh['advanced.unsaved']='未保存的分配';
// Explicit Titanos workflow/control strings; external content is never translated.
const workflowText={
  "ui.select_model": [
    "No model selected.",
    "Ningún modelo seleccionado.",
    "モデルが選択されていません。",
    "未选择模型。"
  ],
  "ui.selected_model": [
    "selected model",
    "modelo seleccionado",
    "選択したモデル",
    "所选模型"
  ],
  "ui.tensor_api": [
    "Tensor models use the typed /api/infer API; Studio does not guess tensor shapes or data.",
    "Los modelos de tensores usan la API tipada /api/infer; Studio no supone formas ni datos.",
    "テンソルモデルは型付き /api/infer API を使用します。Studio は形状やデータを推測しません。",
    "张量模型使用带类型的 /api/infer API；Studio 不会猜测张量形状或数据。"
  ],
  "ui.controls_unavailable": [
    "Model controls unavailable. Refresh to retry. ",
    "Controles del modelo no disponibles. Actualiza para reintentar. ",
    "モデル設定を取得できません。更新して再試行してください。 ",
    "模型控件不可用。请刷新后重试。 "
  ],
  "ui.default_model_server": [
    "Model / server default",
    "Predeterminado del modelo / servidor",
    "モデル / サーバーの既定値",
    "模型 / 服务器默认值"
  ],
  "ui.default_model": [
    "Model default",
    "Predeterminado del modelo",
    "モデルの既定値",
    "模型默认值"
  ],
  "ui.default_pipeline": [
    "Model / pipeline default",
    "Predeterminado del modelo / pipeline",
    "モデル / パイプラインの既定値",
    "模型 / 管线默认值"
  ],
  "ui.default_runtime": [
    "Runtime default",
    "Predeterminado del entorno",
    "実行環境の既定値",
    "运行时默认值"
  ],
  "ui.default": [
    "Default",
    "Predeterminado",
    "既定値",
    "默认值"
  ],
  "ui.default_off": [
    "Default (off)",
    "Predeterminado (desactivado)",
    "既定値（オフ）",
    "默认值（关闭）"
  ],
  "ui.default_filter": [
    "Default (when filtering)",
    "Predeterminado (al filtrar)",
    "既定値（フィルター使用時）",
    "默认值（启用筛选时）"
  ],
  "ui.random": [
    "Random",
    "Aleatorio",
    "ランダム",
    "随机"
  ],
  "ui.on": [
    "On",
    "Activado",
    "オン",
    "开启"
  ],
  "ui.off": [
    "Off",
    "Desactivado",
    "オフ",
    "关闭"
  ],
  "ui.yes": [
    "yes",
    "sí",
    "はい",
    "是"
  ],
  "ui.no": [
    "no",
    "no",
    "いいえ",
    "否"
  ],
  "ui.none": [
    "none",
    "ninguno",
    "なし",
    "无"
  ],
  "ui.unknown": [
    "unknown",
    "desconocido",
    "不明",
    "未知"
  ],
  "ui.default_for_task": [
    "Default for task",
    "Predeterminado para la tarea",
    "タスクの既定値",
    "任务默认值"
  ],
  "ui.disable_request": [
    "Disable for this request",
    "Desactivar para esta solicitud",
    "このリクエストでは無効",
    "对此请求禁用"
  ],
  "ui.whisper_direct": [
    "Whisper direct translation",
    "Traducción directa con Whisper",
    "Whisper による直接翻訳",
    "Whisper 直接翻译"
  ],
  "ui.ocr_settings": [
    "OCR settings",
    "Ajustes de OCR",
    "OCR 設定",
    "OCR 设置"
  ],
  "ui.edit_settings": [
    "Edit settings",
    "Ajustes de edición",
    "編集設定",
    "编辑设置"
  ],
  "ui.native_ocr": [
    "Native OCR · Tesseract",
    "OCR nativo · Tesseract",
    "ネイティブ OCR · Tesseract",
    "原生 OCR · Tesseract"
  ],
  "ui.instruction_edit": [
    "Instruction image edit",
    "Edición de imagen por instrucciones",
    "指示による画像編集",
    "指令式图像编辑"
  ],
  "ui.source_required": [
    "Source image (required)",
    "Imagen de origen (obligatoria)",
    "元画像（必須）",
    "源图像（必填）"
  ],
  "ui.source_optional": [
    "Source image (optional)",
    "Imagen de origen (opcional)",
    "元画像（任意）",
    "源图像（可选）"
  ],
  "ui.source_instruction_help": [
    "Choose the source image and describe the change. Its dimensions are kept.",
    "Elige la imagen de origen y describe el cambio. Se conservan sus dimensiones.",
    "元画像を選び、変更内容を説明してください。画像サイズは維持されます。",
    "请选择源图像并描述修改内容。原图尺寸会保留。"
  ],
  "ui.source_edit_help": [
    "Add a source image to edit it. Leave it empty to generate a new image.",
    "Añade una imagen para editarla. Déjalo vacío para generar una nueva.",
    "編集する元画像を追加してください。新規生成する場合は空欄にします。",
    "添加源图像以进行编辑；留空则生成新图像。"
  ],
  "ui.clear_source": [
    "Clear source image",
    "Quitar imagen de origen",
    "元画像をクリア",
    "清除源图像"
  ],
  "ui.clear_mask": [
    "Clear source and mask",
    "Quitar origen y máscara",
    "元画像とマスクをクリア",
    "清除源图像和蒙版"
  ],
  "ui.source_metadata_empty": [
    "No additional source or runtime metadata reported.",
    "No se recibieron más metadatos de origen o del entorno.",
    "追加の出典・実行環境情報はありません。",
    "未报告其他来源或运行时元数据。"
  ],
  "ui.capabilities_unavailable": [
    "Capabilities unavailable: ",
    "Capacidades no disponibles: ",
    "機能情報を取得できません: ",
    "功能信息不可用："
  ],
  "ui.capabilities": [
    "Capabilities: {{items}}",
    "Capacidades: {{items}}",
    "機能: {{items}}",
    "功能：{{items}}"
  ],
  "ui.request_in_progress": [
    " · request in progress",
    " · solicitud en curso",
    " · リクエスト実行中",
    " · 请求进行中"
  ],
  "ui.asset_image": [
    "Generated image asset",
    "Imagen generada",
    "生成画像",
    "生成的图像资源"
  ],
  "ui.asset_video": [
    "Generated video asset",
    "Vídeo generado",
    "生成動画",
    "生成的视频资源"
  ],
  "ui.asset_audio": [
    "Generated audio asset",
    "Audio generado",
    "生成音声",
    "生成的音频资源"
  ],
  "ui.generated_image": [
    "Generated image",
    "Imagen generada",
    "生成画像",
    "生成的图像"
  ],
  "ui.save_asset": [
    "Save asset",
    "Guardar recurso",
    "アセットを保存",
    "保存资源"
  ],
  "ui.full_result": [
    "Full result and provenance",
    "Resultado completo y procedencia",
    "完全な結果と出典",
    "完整结果与来源"
  ],
  "ui.full_vectors": [
    "Full vectors, usage and input accounting",
    "Vectores completos, uso y datos de entrada",
    "全ベクトル、使用量、入力内訳",
    "完整向量、用量和输入统计"
  ],
  "ui.no_scores": [
    "No numeric scores returned",
    "No se devolvieron puntuaciones numéricas",
    "数値スコアは返されませんでした",
    "未返回数值分数"
  ],
  "ui.uncalibrated": [
    "uncalibrated",
    "sin calibrar",
    "未校正",
    "未经校准"
  ],
  "ui.rerank_note": [
    "Reranker order and raw cross-encoder logits; scores are uncalibrated and are not probabilities.",
    "Orden del reranker y logits brutos del codificador cruzado; no están calibrados ni son probabilidades.",
    "再順位付けとクロスエンコーダーの生ロジットです。スコアは未校正で、確率ではありません。",
    "重排顺序与交叉编码器原始逻辑值；分数未经校准，也不是概率。"
  ],
  "ui.full_rerank": [
    "Full rerank result, input accounting and provenance",
    "Resultado completo de reordenación, entrada y procedencia",
    "再順位付けの全結果、入力内訳、出典",
    "完整重排结果、输入统计和来源"
  ],
  "ui.speech_activity": [
    "speech activity",
    "actividad de voz",
    "音声活動",
    "语音活动"
  ],
  "ui.depth_note": [
    "Relative inverse depth preview: brighter values are nearer. Display normalization is not metric depth and values are not physically comparable across scenes.",
    "Vista de profundidad inversa relativa: los valores claros están más cerca. La normalización visual no indica profundidad métrica ni permite comparar escenas físicamente.",
    "相対逆深度のプレビューです。明るいほど近くを示します。表示上の正規化は実距離ではなく、異なる場面を物理的に比較できません。",
    "相对逆深度预览：越亮表示越近。显示归一化不代表实际距离，不同场景之间的数值不能进行物理比较。"
  ],
  "ui.depth_alt": [
    "Display-normalized relative inverse depth; brighter is nearer",
    "Profundidad inversa relativa normalizada; más claro significa más cerca",
    "表示用に正規化した相対逆深度。明るいほど近い",
    "显示归一化的相对逆深度；越亮表示越近"
  ],
  "ui.save_depth": [
    "Save raw little-endian float32 depth",
    "Guardar profundidad float32 bruta little-endian",
    "生のリトルエンディアン float32 深度を保存",
    "保存小端 float32 原始深度数据"
  ],
  "ui.depth_details": [
    "Depth metadata and provenance",
    "Metadatos y procedencia de profundidad",
    "深度のメタデータと出典",
    "深度元数据与来源"
  ],
  "ui.mask_note": [
    "Image-aligned RGBA edit masks in native order. Predicted IoU values are uncalibrated; Studio does not choose a “best” mask or invent class labels.",
    "Máscaras RGBA alineadas con la imagen, en orden nativo. El IoU previsto no está calibrado; Studio no elige la «mejor» máscara ni inventa etiquetas.",
    "画像に整合した RGBA マスクをネイティブ順で表示します。予測 IoU は未校正です。Studio は「最良」のマスクを選んだりラベルを創作したりしません。",
    "按原生顺序显示与图像对齐的 RGBA 蒙版。预测 IoU 未经校准；Studio 不会选择“最佳”蒙版或编造类别标签。"
  ],
  "ui.save_mask": [
    "Save mask PNG",
    "Guardar máscara PNG",
    "マスク PNG を保存",
    "保存蒙版 PNG"
  ],
  "ui.mask_details": [
    "Mask prompt, native order and provenance",
    "Indicación de máscara, orden nativo y procedencia",
    "マスクの指示、ネイティブ順、出典",
    "蒙版提示、原生顺序和来源"
  ],
  "ui.no_text": [
    "No text returned.",
    "No se devolvió texto.",
    "テキストは返されませんでした。",
    "未返回文本。"
  ],
  "ui.transcript_details": [
    "Transcript segments, quality fields and provenance",
    "Segmentos de transcripción, calidad y procedencia",
    "書き起こし区間、品質情報、出典",
    "转录分段、质量字段和来源"
  ],
  "ui.source_unspecified": [
    "source unspecified",
    "origen no indicado",
    "翻訳元未指定",
    "未指定源语言"
  ],
  "ui.target_unspecified": [
    "target unspecified",
    "destino no indicado",
    "翻訳先未指定",
    "未指定目标语言"
  ],
  "ui.ocr_details": [
    "OCR details and geometry (native confidence is uncalibrated)",
    "Detalles y geometría OCR (confianza nativa sin calibrar)",
    "OCR 詳細と位置情報（ネイティブ信頼度は未校正）",
    "OCR 详情和几何信息（原生置信度未经校准）"
  ],
  "ui.music_details": [
    "Music format, effective controls and provenance",
    "Formato musical, controles efectivos y procedencia",
    "音楽形式、実効設定、出典",
    "音乐格式、实际控制参数和来源"
  ],
  "ui.preserve_details": [
    "Mask and pixel preservation details",
    "Detalles de máscara y conservación de píxeles",
    "マスクと画素保持の詳細",
    "蒙版与像素保留详情"
  ],
  "ui.pair_note": [
    "Model-native predictions. Scores are uncalibrated softmax values, not verified truth or calibrated probabilities.",
    "Predicciones nativas del modelo. Las puntuaciones softmax no están calibradas ni son hechos verificados o probabilidades calibradas.",
    "モデル本来の予測です。スコアは未校正の softmax 値で、検証済みの事実や校正された確率ではありません。",
    "模型原生预测。分数是未经校准的 softmax 值，并非已验证事实或经校准的概率。"
  ],
  "ui.class_outputs": [
    "Native class outputs",
    "Resultados de clases nativas",
    "ネイティブクラス出力",
    "原生类别输出"
  ],
  "ui.class_index": [
    "Class index",
    "Índice de clase",
    "クラス番号",
    "类别索引"
  ],
  "ui.native_label": [
    "Native label",
    "Etiqueta nativa",
    "ネイティブラベル",
    "原生标签"
  ],
  "ui.raw_logit": [
    "Raw logit",
    "Logit bruto",
    "生ロジット",
    "原始逻辑值"
  ],
  "ui.raw_score": [
    "Uncalibrated score",
    "Puntuación sin calibrar",
    "未校正スコア",
    "未经校准的分数"
  ],
  "ui.pair_details": [
    "Full result, input accounting and model provenance",
    "Resultado completo, entrada y procedencia del modelo",
    "全結果、入力内訳、モデルの出典",
    "完整结果、输入统计和模型来源"
  ],
  "ui.generating": [
    "Generating…",
    "Generando…",
    "生成中…",
    "正在生成…"
  ],
  "ui.job_queued": [
    "Queued. The job is listed under Jobs and keeps running if you leave this page.",
    "En cola. El trabajo aparece en Trabajos y sigue ejecutándose si sales de esta página.",
    "待機中。このジョブは「ジョブ」に表示され、ページを離れても実行を続けます。",
    "已排队。该任务列在“任务”中，离开此页后仍会继续运行。"
  ],
  "ui.stopped": [
    "Stopped.",
    "Detenido.",
    "停止しました。",
    "已停止。"
  ],
  "ui.cancel_failed": [
    "Job cancellation failed: ",
    "No se pudo cancelar el trabajo: ",
    "ジョブをキャンセルできませんでした: ",
    "任务取消失败："
  ],
  "ui.registration_name": [
    "Name",
    "Nombre",
    "名前",
    "名称"
  ],
  "ui.registration_path": [
    "Checkpoint folder",
    "Carpeta del modelo",
    "チェックポイントフォルダー",
    "模型权重文件夹"
  ],
  "ui.registration_task": [
    "Task",
    "Tarea",
    "タスク",
    "任务"
  ],
  "ui.registration_backend": [
    "Backend",
    "Motor",
    "バックエンド",
    "后端"
  ],
  "ui.registration_device": [
    "Device",
    "Dispositivo",
    "デバイス",
    "设备"
  ],
  "ui.registration_variant": [
    "Weight variant (optional)",
    "Variante de pesos (opcional)",
    "重みの種類（任意）",
    "权重变体（可选）"
  ],
  "ui.dtype": [
    "Compute dtype",
    "Tipo de cálculo",
    "計算データ型",
    "计算数据类型"
  ],
  "ui.graph": [
    "Prepared graph filename (relative, optional)",
    "Archivo de grafo preparado (relativo, opcional)",
    "準備済みグラフのファイル名（相対パス、任意）",
    "已准备图文件名（相对路径，可选）"
  ],
  "ui.graph_example": [
    "e.g. model.onnx or model.xml",
    "p. ej. model.onnx o model.xml",
    "例: model.onnx または model.xml",
    "例如 model.onnx 或 model.xml"
  ],
  "ui.voice_map": [
    "Voice map JSON (optional)",
    "Mapa de voces JSON (opcional)",
    "音声マップ JSON（任意）",
    "音色映射 JSON（可选）"
  ],
  "ui.dml_adapter": [
    "DirectML adapter",
    "Adaptador DirectML",
    "DirectML アダプター",
    "DirectML 适配器"
  ],
  "ui.load_devices_first": [
    "Load known devices first",
    "Carga primero los dispositivos conocidos",
    "先に既知のデバイスを読み込んでください",
    "请先加载已知设备"
  ],
  "ui.dml_metacommands": [
    "DirectML metacommands",
    "Metacomandos DirectML",
    "DirectML メタコマンド",
    "DirectML 元命令"
  ],
  "ui.enabled_false": [
    "Enabled (explicit false)",
    "Activados (false explícito)",
    "有効（明示的な false）",
    "已启用（显式 false）"
  ],
  "ui.disabled_diagnostic": [
    "Disabled (diagnostic only)",
    "Desactivados (solo diagnóstico)",
    "無効（診断専用）",
    "已禁用（仅用于诊断）"
  ],
  "ui.replace_registration": [
    "Replace an existing registration",
    "Reemplazar un registro existente",
    "既存の登録を置換",
    "替换现有注册"
  ],
  "ui.load_devices": [
    "Load known devices",
    "Cargar dispositivos conocidos",
    "既知のデバイスを読み込む",
    "加载已知设备"
  ],
  "ui.add_model": [
    "Add model",
    "Añadir modelo",
    "モデルを追加",
    "添加模型"
  ],
  "ui.observed_runtime": [
    "Observed loaded runtime details",
    "Detalles observados del entorno cargado",
    "ロード済み実行環境の観測詳細",
    "已加载运行时的观测详情"
  ],
  "ui.require_loaded": [
    "Require this chat model to be loaded; do not start it implicitly",
    "Exigir que este modelo de chat esté cargado; no iniciarlo automáticamente",
    "このチャットモデルの事前ロードを必須にし、暗黙に起動しない",
    "要求此对话模型已加载；不要隐式启动"
  ],
  "ui.wrong_assignment": [
    "Server returned an assignment for a different model.",
    "El servidor devolvió una asignación de otro modelo.",
    "サーバーが別のモデルの割り当てを返しました。",
    "服务器返回了其他模型的分配。"
  ],
  "ui.replica_device": [
    "Choose one stable device for each replica.",
    "Elige un dispositivo estable para cada réplica.",
    "各レプリカに一つの安定したデバイスを選んでください。",
    "请为每个副本选择一个具有稳定标识的设备。"
  ],
  "ui.registration_complete": [
    "Model registered. Its external checkpoint files were not copied or changed.",
    "Modelo registrado. Sus archivos externos no se copiaron ni modificaron.",
    "モデルを登録しました。外部の重みファイルはコピー・変更していません。",
    "模型已注册。外部权重文件未被复制或修改。"
  ],
  "ui.new_conversation": [
    "New conversation.",
    "Nueva conversación.",
    "新しい会話です。",
    "已开始新对话。"
  ],
  "ui.loading_model": [
    "Loading model…",
    "Cargando modelo…",
    "モデルをロード中…",
    "正在加载模型…"
  ],
  "ui.model_loaded": [
    "Model loaded. Runtime details are now available.",
    "Modelo cargado. Ya puedes consultar los detalles del entorno.",
    "モデルをロードしました。実行環境の詳細を確認できます。",
    "模型已加载。现在可以查看运行时详情。"
  ],
  "ui.model_unloaded": [
    "Model unloaded.",
    "Modelo liberado de memoria.",
    "モデルをアンロードしました。",
    "模型已卸载。"
  ],
  "ui.showing_runtime": [
    "Showing observed runtime details for the loaded model.",
    "Mostrando los detalles observados del entorno del modelo cargado.",
    "ロード済みモデルの実行環境の観測詳細を表示しています。",
    "正在显示已加载模型的运行时观测详情。"
  ],
  "ui.registration_removed": [
    "Model registration removed. External checkpoint and voice files were not deleted.",
    "Registro eliminado. No se borraron los archivos externos de pesos ni de voz.",
    "モデル登録を削除しました。外部の重み・音声ファイルは削除していません。",
    "模型注册已删除。外部权重和音色文件未被删除。"
  ],
  "ui.runtime_profile": [
    "Managed native runtime (GGUF chat)",
    "Entorno nativo gestionado (chat GGUF)",
    "管理対象ネイティブ実行環境（GGUF チャット）",
    "托管原生运行时（GGUF 对话）"
  ],
  "ui.search_example": [
    "small English embedding model",
    "modelo pequeño de embeddings en inglés",
    "小型の英語埋め込みモデル",
    "小型英语嵌入模型"
  ],
  "ui.gated_access": [
    "Gated access",
    "Acceso restringido",
    "アクセス制限あり",
    "受限访问"
  ],
  "ui.public_access": [
    "Public access",
    "Acceso público",
    "公開アクセス",
    "公开访问"
  ],
  "ui.compatibility_note": [
    "Compatibility is verified during planning and load.",
    "La compatibilidad se comprueba al planificar y cargar.",
    "互換性は計画時とロード時に確認します。",
    "兼容性会在规划和加载时验证。"
  ],
  "ui.searching": [
    "Searching provider metadata…",
    "Buscando metadatos del proveedor…",
    "提供元のメタデータを検索中…",
    "正在搜索提供方元数据…"
  ],
  "ui.no_candidates": [
    "No matching candidates. Try a broader query or another source.",
    "No hay coincidencias. Prueba una búsqueda más amplia u otra fuente.",
    "候補が見つかりません。検索条件を広げるか、別の提供元を試してください。",
    "没有匹配的候选模型。请扩大搜索范围或选择其他来源。"
  ],
  "ui.enter_link": [
    "Enter an HTTPS model page or artifact link.",
    "Introduce un enlace HTTPS a una página o archivo de modelo.",
    "モデルページまたはファイルの HTTPS リンクを入力してください。",
    "请输入 HTTPS 模型页面或资源链接。"
  ],
  "ui.inspecting_link": [
    "Inspecting the model link…",
    "Inspeccionando el enlace del modelo…",
    "モデルリンクを確認中…",
    "正在检查模型链接…"
  ],
  "ui.recognized_provider": [
    "Recognized provider metadata. Compatibility still requires a concrete plan.",
    "Proveedor reconocido. La compatibilidad aún requiere un plan concreto.",
    "提供元を認識しました。互換性の確認には具体的な計画が必要です。",
    "已识别提供方元数据。仍需制定具体计划以验证兼容性。"
  ],
  "ui.public_page": [
    "Public model page",
    "Página pública del modelo",
    "公開モデルページ",
    "公开模型页面"
  ],
  "ui.no_description": [
    "No page description was declared.",
    "La página no declara una descripción.",
    "ページの説明はありません。",
    "页面未提供说明。"
  ],
  "ui.not_declared": [
    "not declared",
    "no declarada",
    "記載なし",
    "未声明"
  ],
  "ui.inspect_provider_link": [
    "Inspect provider link",
    "Inspeccionar enlace del proveedor",
    "提供元リンクを確認",
    "检查提供方链接"
  ],
  "ui.page_evidence": [
    "Technical page evidence",
    "Datos técnicos de la página",
    "ページの技術的根拠",
    "页面技术证据"
  ],
  "ui.page_imported": [
    "Public page metadata imported. Linked providers still require inspect and planning.",
    "Metadatos importados. Los proveedores enlazados aún requieren inspección y planificación.",
    "公開ページのメタデータを取り込みました。リンク先の提供元は確認と計画が必要です。",
    "已导入公开页面元数据。链接的提供方仍需检查并制定计划。"
  ],
  "ui.metadata_link": [
    "Metadata-only model link",
    "Enlace de modelo: solo metadatos",
    "メタデータのみのモデルリンク",
    "仅元数据模型链接"
  ],
  "ui.metadata_only_note": [
    "No supported provider/format was resolved. Keep this link as provenance; Titanos will not execute page instructions or claim compatibility.",
    "No se identificó un proveedor o formato compatible. Conserva el enlace como procedencia; Titanos no ejecutará instrucciones de la página ni afirmará compatibilidad.",
    "対応する提供元・形式を特定できません。出典としてリンクを保持します。Titanos はページの指示を実行せず、互換性を保証しません。",
    "未解析到受支持的提供方或格式。链接仅作为来源记录；Titanos 不会执行页面指令或宣称兼容。"
  ],
  "ui.metadata_link_imported": [
    "Link imported as metadata only.",
    "Enlace importado solo como metadatos.",
    "リンクをメタデータとして取り込みました。",
    "链接已作为元数据导入。"
  ],
  "ui.resolving_plan": [
    "Resolving an immutable acquisition plan…",
    "Resolviendo un plan de adquisición inmutable…",
    "固定された取得計画を作成中…",
    "正在确定不可变的获取计划…"
  ],
  "ui.no_gguf": [
    "No compatible GGUF model file was found in this exact revision.",
    "No se encontró un archivo GGUF compatible en esta revisión exacta.",
    "この正確なリビジョンに互換 GGUF モデルがありません。",
    "在此确切修订版本中未找到兼容的 GGUF 模型文件。"
  ],
  "ui.missing_projector": [
    "Native vision requires a matching GGUF projector in the same revision.",
    "La visión nativa requiere un proyector GGUF compatible en la misma revisión.",
    "ネイティブ画像理解には、同じリビジョンの対応 GGUF プロジェクターが必要です。",
    "原生视觉需要同一修订版本中的匹配 GGUF 投影器。"
  ],
  "ui.size_unavailable": [
    "Size unavailable",
    "Tamaño no disponible",
    "容量不明",
    "大小不可用"
  ],
  "ui.license_review_needed": [
    "Inspect acquired license before activation",
    "Revisa la licencia adquirida antes de activar",
    "有効化する前に取得したライセンスを確認",
    "激活前请检查已获取的许可"
  ],
  "ui.unmeasured": [
    "not measured for this exact model/device",
    "no medido para este modelo/dispositivo exacto",
    "このモデル・デバイスの組み合わせでは未測定",
    "尚未测量此确切模型与设备组合"
  ],
  "ui.prepare_then_run": [
    "Prepare CPU runtime, then download and run",
    "Preparar el entorno CPU, descargar y ejecutar",
    "CPU 実行環境を準備してダウンロード・実行",
    "准备 CPU 运行时，然后下载并运行"
  ],
  "ui.plan_ready": [
    "Plan resolved. Downloading will not activate a partial or failed model.",
    "Plan listo. La descarga no activará un modelo incompleto o fallido.",
    "計画を確定しました。不完全または失敗したモデルを有効化することはありません。",
    "计划已确定。下载不会激活不完整或失败的模型。"
  ],
  "ui.acquisition_complete_prefix": [
    "Acquisition complete:",
    "Adquisición completada:",
    "取得完了:",
    "获取完成："
  ],
  "ui.preparing": [
    "Preparing",
    "Preparando",
    "準備中",
    "正在准备"
  ],
  "ui.cancel_requested": [
    "Cancellation requested. Partial data remains inactive.",
    "Cancelación solicitada. Los datos parciales siguen inactivos.",
    "キャンセルを要求しました。不完全なデータは無効のままです。",
    "已请求取消。部分数据仍处于未激活状态。"
  ],
  "ui.preparing_runtime": [
    "Preparing managed CPU runtime",
    "Preparando el entorno CPU gestionado",
    "管理対象 CPU 実行環境を準備中",
    "正在准备托管 CPU 运行时"
  ],
  "ui.cancel_runtime": [
    "Cancel runtime setup",
    "Cancelar preparación del entorno",
    "実行環境の準備をキャンセル",
    "取消运行时准备"
  ],
  "ui.runtime_ready": [
    "Managed CPU runtime is verified and ready.",
    "El entorno CPU gestionado está verificado y listo.",
    "管理対象 CPU 実行環境を検証し、準備ができました。",
    "托管 CPU 运行时已验证并准备就绪。"
  ],
  "ui.retry_runtime": [
    "Retry runtime setup",
    "Reintentar preparación del entorno",
    "実行環境の準備を再試行",
    "重试运行时准备"
  ],
  "ui.runtime_missing": [
    "This server has no compatible operator-configured native runtime profile.",
    "Este servidor no tiene un perfil nativo compatible configurado por el operador.",
    "このサーバーに管理者設定の互換ネイティブ実行環境プロファイルがありません。",
    "此服务器没有由管理员配置的兼容原生运行时配置。"
  ],
  "ui.planning_runtime": [
    "Planning the server-owned CPU runtime…",
    "Planificando el entorno CPU del servidor…",
    "サーバー管理の CPU 実行環境を計画中…",
    "正在规划服务器管理的 CPU 运行时…"
  ],
  "ui.unexpected_runtime": [
    "Server returned an unexpected runtime plan.",
    "El servidor devolvió un plan de entorno inesperado.",
    "サーバーが予期しない実行環境計画を返しました。",
    "服务器返回了意外的运行时计划。"
  ],
  "ui.review_all_licenses": [
    "Review all acquired model license evidence before activation.",
    "Revisa todas las pruebas de licencia del modelo antes de activarlo.",
    "有効化する前に、取得したすべてのモデルライセンス資料を確認してください。",
    "激活前请检查所有已获取的模型许可资料。"
  ],
  "ui.licenses_ack": [
    "I reviewed every listed license/provenance document",
    "He revisado todos los documentos de licencia y procedencia indicados",
    "記載されたすべてのライセンス・出典資料を確認しました",
    "我已阅读列出的所有许可和来源文件"
  ],
  "ui.activate_reviewed": [
    "Activate after review",
    "Activar tras revisar",
    "確認して有効化",
    "检查后激活"
  ],
  "ui.ack_required": [
    "Review and check the complete license acknowledgement first.",
    "Primero revisa y marca la aceptación de todas las licencias.",
    "先にすべてのライセンスを確認し、確認欄にチェックしてください。",
    "请先阅读全部许可并勾选确认。"
  ],
  "ui.registering_loading": [
    "Registering and loading the verified model…",
    "Registrando y cargando el modelo verificado…",
    "検証済みモデルを登録・ロード中…",
    "正在注册并加载已验证的模型…"
  ],
  "ui.active_catalog_failed": [
    "This acquisition is complete, but the model catalog could not be refreshed. Refresh Models before selecting it.",
    "La adquisición terminó, pero no se pudo actualizar el catálogo. Actualiza Modelos antes de seleccionarlo.",
    "取得は完了しましたがカタログを更新できませんでした。選択する前にモデル一覧を更新してください。",
    "获取已完成，但模型目录刷新失败。选择模型前请刷新“模型”。"
  ],
  "ui.active_registration_changed": [
    "This acquisition is complete, but its model registration is missing or has changed. Review Models before running it.",
    "La adquisición terminó, pero falta el registro o ha cambiado. Revisa Modelos antes de ejecutarlo.",
    "取得は完了しましたが、登録が見つからないか変更されています。実行前にモデル一覧を確認してください。",
    "获取已完成，但模型注册缺失或已更改。运行前请检查“模型”。"
  ],
  "ui.already_registered": [
    "This model is already registered. Continue in the task workspace below.",
    "Este modelo ya está registrado. Continúa en el espacio de trabajo de abajo.",
    "このモデルは登録済みです。下の作業欄に進んでください。",
    "此模型已注册。请在下方工作区继续。"
  ],
  "ui.acquisition_busy": [
    "Another model acquisition is already active in this tab.",
    "Ya hay otra adquisición de modelo activa en esta pestaña.",
    "このタブでは別のモデル取得が実行中です。",
    "此标签页已有其他模型获取任务正在进行。"
  ],
  "ui.acquisition_starting": [
    "Starting durable acquisition…",
    "Iniciando adquisición recuperable…",
    "再開可能な取得を開始中…",
    "正在启动可恢复的获取任务…"
  ],
  "ui.retry_acquisition": [
    "Retry / resume",
    "Reintentar / continuar",
    "再試行 / 再開",
    "重试 / 继续"
  ],
  "ui.tesseract_help": [
    "Choose the folder containing LSTM .traineddata files. Tesseract uses the CPU.",
    "Elige la carpeta con archivos LSTM .traineddata. Tesseract usa la CPU.",
    "LSTM .traineddata ファイルのフォルダーを選んでください。Tesseract は CPU を使用します。",
    "请选择包含 LSTM .traineddata 文件的文件夹。Tesseract 使用 CPU。"
  ],
  "ui.vad_help": [
    "Choose a Silero-compatible ONNX graph. VAD is CPU-only.",
    "Elige un grafo ONNX compatible con Silero. VAD solo usa CPU.",
    "Silero 互換 ONNX グラフを選んでください。VAD は CPU 専用です。",
    "请选择兼容 Silero 的 ONNX 图。VAD 仅使用 CPU。"
  ],
  "ui.onnx_image_help": [
    "Choose a prepared SD1 ONNX component directory. DirectML needs a current adapter index and LUID; CPU needs neither.",
    "Elige un directorio preparado de componentes SD1 ONNX. DirectML necesita un índice y LUID actuales; CPU no los necesita.",
    "準備済み SD1 ONNX コンポーネントを選んでください。DirectML は現在のアダプター番号と LUID が必要ですが、CPU は不要です。",
    "请选择已准备的 SD1 ONNX 组件目录。DirectML 需要当前适配器索引和 LUID；CPU 不需要。"
  ],
  "ui.checkpoint_help": [
    "Point to a complete local checkpoint with its tokenizer, processor, components or prepared graph. Model weights stay outside Titanos.",
    "Selecciona un modelo local completo con tokenizer, procesador, componentes o grafo preparado. Los pesos permanecen fuera de Titanos.",
    "トークナイザー、プロセッサー、コンポーネントや準備済みグラフを含む完全なローカルモデルを指定してください。重みは Titanos の外部に保持されます。",
    "请选择包含分词器、处理器、组件或已准备图的完整本地模型。权重始终保存在 Titanos 外部。"
  ],
  "ui.dml_choose": [
    "Choose a current DirectML adapter",
    "Elige un adaptador DirectML actual",
    "現在の DirectML アダプターを選択",
    "选择当前 DirectML 适配器"
  ],
  "ui.devices_empty": [
    "No cached devices. Enter a supported runtime address or refresh inventory with the CLI.",
    "No hay dispositivos en caché. Introduce una dirección admitida o actualiza el inventario mediante la CLI.",
    "保存済みデバイスがありません。対応アドレスを入力するか CLI で一覧を更新してください。",
    "没有缓存的设备。请输入受支持的运行时地址，或通过 CLI 刷新清单。"
  ],
  "ui.devices_unavailable": [
    "Device inventory unavailable; manual device entry remains enabled. ",
    "Inventario no disponible; puedes introducir el dispositivo manualmente. ",
    "デバイス一覧を取得できません。手動入力は利用できます。 ",
    "设备清单不可用；仍可手动输入设备。 "
  ],
  "ui.native_registration_help": [
    "llama-cpp needs native GGUF options. Register it with titanos add --backend llama-cpp.",
    "llama-cpp necesita opciones GGUF nativas. Regístralo con titanos add --backend llama-cpp.",
    "llama-cpp にはネイティブ GGUF 設定が必要です。titanos add --backend llama-cpp で登録してください。",
    "llama-cpp 需要原生 GGUF 选项。请使用 titanos add --backend llama-cpp 注册。"
  ],
  "ui.xtts_registration_help": [
    "XTTS requires explicit trusted pickle permission. Register it with titanos add --backend xtts --allow-pickle.",
    "XTTS requiere permiso explícito para pickle de confianza. Regístralo con titanos add --backend xtts --allow-pickle.",
    "XTTS には信頼した pickle の明示的な許可が必要です。titanos add --backend xtts --allow-pickle で登録してください。",
    "XTTS 需要明确允许受信任的 pickle。请使用 titanos add --backend xtts --allow-pickle 注册。"
  ],
  "ui.found_candidates": [
    "Found {{count}} candidates. Review compatibility before downloading.",
    "Se encontraron {{count}} candidatos. Revisa su compatibilidad antes de descargar.",
    "候補が {{count}} 件見つかりました。ダウンロード前に互換性を確認してください。",
    "找到 {{count}} 个候选模型。下载前请检查兼容性。"
  ],
  "ui.get_ready": [
    "Get {{model}} ready",
    "Preparar {{model}}",
    "{{model}} を準備",
    "准备 {{model}}"
  ],
  "ui.pinned_revision": [
    "Pinned revision: {{revision}}",
    "Revisión fijada: {{revision}}",
    "固定リビジョン: {{revision}}",
    "固定修订版本：{{revision}}"
  ],
  "ui.license_access": [
    "License: {{license}} · Access: {{access}}",
    "Licencia: {{license}} · Acceso: {{access}}",
    "ライセンス: {{license}} · アクセス: {{access}}",
    "许可：{{license}} · 访问：{{access}}"
  ],
  "ui.runtime_qualification": [
    "Runtime: {{runtime}} · Qualification: {{qualification}}",
    "Entorno: {{runtime}} · Validación: {{qualification}}",
    "実行環境: {{runtime}} · 検証: {{qualification}}",
    "运行时：{{runtime}} · 验证状态：{{qualification}}"
  ],
  "ui.selected_files": [
    "{{count}} selected files",
    "{{count}} archivos seleccionados",
    "選択したファイル {{count}} 件",
    "已选择 {{count}} 个文件"
  ],
  "ui.page_license": [
    "Page-declared license: {{license}} · Compatibility: unverified",
    "Licencia declarada: {{license}} · Compatibilidad: sin verificar",
    "ページ記載のライセンス: {{license}} · 互換性: 未検証",
    "页面声明的许可：{{license}} · 兼容性：未验证"
  ],
  "ui.acquisition_heading": [
    "{{state}} {{model}}",
    "{{state}} {{model}}",
    "{{state}} {{model}}",
    "{{state}} {{model}}"
  ],
  "ui.progress": [
    "{{status}} · {{stage}} · {{completed}}{{total}}",
    "{{status}} · {{stage}} · {{completed}}{{total}}",
    "{{status}} · {{stage}} · {{completed}}{{total}}",
    "{{status}} · {{stage}} · {{completed}}{{total}}"
  ],
  "ui.progress_total": [
    " / {{total}} {{unit}}",
    " / {{total}} {{unit}}",
    " / {{total}} {{unit}}",
    " / {{total}} {{unit}}"
  ],
  "unit.bytes": [
    "bytes",
    "bytes",
    "バイト",
    "字节"
  ],
  "unit.files": [
    "files",
    "archivos",
    "ファイル",
    "文件"
  ],
  "unit.steps": [
    "steps",
    "pasos",
    "ステップ",
    "步骤"
  ],
  "ui.no_python": [
    "no user Python required",
    "no requiere Python del usuario",
    "Python の別途インストール不要",
    "用户无需安装 Python"
  ],
  "ui.runtime_summary": [
    "{{runtime}} {{release}} · {{profile}} · {{license}} · {{note}}",
    "{{runtime}} {{release}} · {{profile}} · {{license}} · {{note}}",
    "{{runtime}} {{release}} · {{profile}} · {{license}} · {{note}}",
    "{{runtime}} {{release}} · {{profile}} · {{license}} · {{note}}"
  ],
  "ui.runtime_state": [
    "Runtime setup {{status}}.",
    "Preparación del entorno: {{status}}.",
    "実行環境の準備: {{status}}。",
    "运行时准备：{{status}}。"
  ],
  "ui.acquisition_state": [
    "Acquisition {{status}}.",
    "Adquisición: {{status}}.",
    "取得: {{status}}。",
    "获取：{{status}}。"
  ],
  "ui.acquisition_recovered": [
    "Recovered {{status}} acquisition {{model}}.",
    "Adquisición recuperada de {{model}}: {{status}}.",
    "{{model}} の取得を復元しました: {{status}}。",
    "已恢复 {{model}} 的获取任务：{{status}}。"
  ],
  "ui.replica_summary": [
    "{{mode}} · {{count}} replicas",
    "{{mode}} · {{count}} réplicas",
    "{{mode}} · レプリカ {{count}} 個",
    "{{mode}} · {{count}} 个副本"
  ],
  "ui.replica_saved": [
    "{{mode}} · {{count}} replicas saved",
    "{{mode}} · {{count}} réplicas guardadas",
    "{{mode}} · レプリカ {{count}} 個を保存",
    "{{mode}} · 已保存 {{count}} 个副本"
  ],
  "ui.devices_loaded": [
    "Loaded {{count}} known device records. Inventory presence is not inference qualification.",
    "Se cargaron {{count}} dispositivos conocidos. Estar en el inventario no acredita la inferencia.",
    "既知のデバイス {{count}} 件を読み込みました。一覧への存在は推論の検証を意味しません。",
    "已加载 {{count}} 条已知设备记录。清单中存在设备并不代表推理已获验证。"
  ],
  "ui.remove_confirm": [
    "Remove the {{model}} registration? External checkpoint and voice files will not be deleted.",
    "¿Eliminar el registro de {{model}}? Los archivos externos de pesos y voces no se borrarán.",
    "{{model}} の登録を削除しますか？外部の重み・音声ファイルは削除されません。",
    "删除 {{model}} 的注册？外部权重和音色文件不会被删除。"
  ],
  "ui.recommended": [
    "recommended",
    "recomendado",
    "推奨",
    "推荐"
  ],
  "ui.job_name": [
    "{{model}} · {{task}}",
    "{{model}} · {{task}}",
    "{{model}} · {{task}}",
    "{{model}} · {{task}}"
  ],
  "ui.job_phase": [
    "{{status}}{{phase}}",
    "{{status}}{{phase}}",
    "{{status}}{{phase}}",
    "{{status}}{{phase}}"
  ],
  "ui.model_title": [
    "{{model}} · {{task}}",
    "{{model}} · {{task}}",
    "{{model}} · {{task}}",
    "{{model}} · {{task}}"
  ],
  "ui.assistant": [
    "Assistant · {{model}}",
    "Asistente · {{model}}",
    "アシスタント · {{model}}",
    "助手 · {{model}}"
  ],
  "ui.you": [
    "You · {{model}}",
    "Tú · {{model}}",
    "あなた · {{model}}",
    "你 · {{model}}"
  ],
  "ui.asset": [
    "Asset",
    "Recurso",
    "アセット",
    "资源"
  ],
  "ui.partial_stopped": [
    "Stopped (partial)",
    "Detenido (parcial)",
    "停止（途中まで）",
    "已停止（部分结果）"
  ],
  "ui.incomplete": [
    "Incomplete",
    "Incompleto",
    "未完了",
    "未完成"
  ],
  "ui.result_title": [
    "{{task}} · {{model}}",
    "{{task}} · {{model}}",
    "{{task}} · {{model}}",
    "{{task}} · {{model}}"
  ],
  "status.loaded": [
    "loaded",
    "cargado",
    "ロード済み",
    "已加载"
  ],
  "status.unloaded": [
    "unloaded",
    "sin cargar",
    "未ロード",
    "未加载"
  ],
  "status.loading": [
    "loading",
    "cargando",
    "ロード中",
    "正在加载"
  ],
  "status.error": [
    "error",
    "error",
    "エラー",
    "错误"
  ],
  "status.active": [
    "active",
    "activo",
    "有効",
    "已激活"
  ],
  "status.prepared": [
    "prepared",
    "preparado",
    "準備済み",
    "已准备"
  ],
  "status.registered": [
    "registered",
    "registrado",
    "登録済み",
    "已注册"
  ],
  "status.available": [
    "available",
    "disponible",
    "利用可能",
    "可用"
  ],
  "status.absent": [
    "absent",
    "ausente",
    "未導入",
    "未安装"
  ],
  "status.missing_runtime": [
    "missing runtime",
    "falta el entorno",
    "実行環境なし",
    "缺少运行时"
  ],
  "status.unavailable": [
    "unavailable",
    "no disponible",
    "利用不可",
    "不可用"
  ],
  "status.unknown": [
    "unknown",
    "desconocido",
    "不明",
    "未知"
  ],
  "stage.downloading": [
    "downloading",
    "descargando",
    "ダウンロード中",
    "正在下载"
  ],
  "stage.downloading_files": [
    "downloading files",
    "descargando archivos",
    "ファイルをダウンロード中",
    "正在下载文件"
  ],
  "stage.verifying": [
    "verifying",
    "verificando",
    "検証中",
    "正在验证"
  ],
  "stage.extracting": [
    "extracting",
    "extrayendo",
    "展開中",
    "正在解压"
  ],
  "stage.preparing": [
    "preparing",
    "preparando",
    "準備中",
    "正在准备"
  ],
  "stage.prepared": [
    "prepared",
    "preparado",
    "準備済み",
    "已准备"
  ],
  "stage.loading": [
    "loading",
    "cargando",
    "ロード中",
    "正在加载"
  ],
  "stage.planning": [
    "planning",
    "planificando",
    "計画中",
    "正在规划"
  ],
  "stage.conditioning": [
    "conditioning",
    "preparando condiciones",
    "条件を準備中",
    "正在准备条件"
  ],
  "stage.sampling": [
    "sampling",
    "generando muestras",
    "サンプリング中",
    "正在采样"
  ],
  "stage.decoding": [
    "decoding",
    "decodificando",
    "デコード中",
    "正在解码"
  ],
  "stage.saving": [
    "saving",
    "guardando",
    "保存中",
    "正在保存"
  ],
  "stage.completed": [
    "completed",
    "completado",
    "完了",
    "已完成"
  ],
  "stage.cancelled": [
    "cancelled",
    "cancelado",
    "キャンセル済み",
    "已取消"
  ],
  "stage.queued": [
    "queued",
    "en cola",
    "待機中",
    "排队中"
  ],
  "ui.documents": [
    "Documents (one nonempty passage per line)",
    "Documentos (un pasaje no vacío por línea)",
    "文書（1 行に空でない文章を一つ）",
    "文档（每行一段非空文本）"
  ],
  "ui.top_n": [
    "Rerank result count",
    "Número de resultados reordenados",
    "再順位付けの結果数",
    "重排结果数量"
  ],
  "ui.rerank_truncate": [
    "Rerank input truncation",
    "Recorte de entrada para reordenación",
    "再順位付け入力の切り詰め",
    "重排输入截断"
  ],
  "ui.reject_oversized": [
    "Reject oversized inputs",
    "Rechazar entradas demasiado grandes",
    "上限超過の入力を拒否",
    "拒绝过大的输入"
  ],
  "ui.allow_truncate": [
    "Allow truncation",
    "Permitir recorte",
    "切り詰めを許可",
    "允许截断"
  ],
  "ui.return_documents": [
    "Return document text",
    "Devolver texto de documentos",
    "文書の本文を返す",
    "返回文档文本"
  ],
  "ui.vad_rate": [
    "VAD sample rate",
    "Frecuencia de muestreo VAD",
    "VAD サンプリング周波数",
    "VAD 采样率"
  ],
  "ui.vad_threshold": [
    "VAD threshold",
    "Umbral VAD",
    "VAD しきい値",
    "VAD 阈值"
  ],
  "ui.vad_min_speech": [
    "Minimum speech (ms)",
    "Voz mínima (ms)",
    "最短音声区間（ms）",
    "最短语音时长（毫秒）"
  ],
  "ui.vad_min_silence": [
    "Minimum silence (ms)",
    "Silencio mínimo (ms)",
    "最短無音区間（ms）",
    "最短静音时长（毫秒）"
  ],
  "ui.vad_padding": [
    "Speech padding (ms)",
    "Margen de voz (ms)",
    "音声区間の余白（ms）",
    "语音区间扩展（毫秒）"
  ],
  "ui.vad_probabilities": [
    "Return frame probabilities",
    "Devolver probabilidades por fotograma",
    "フレーム確率を返す",
    "返回帧概率"
  ],
  "ui.segmentation_prompt": [
    "Segmentation prompt",
    "Indicación de segmentación",
    "領域分割の指示",
    "分割提示"
  ],
  "ui.box_coordinates": [
    "Box coordinates",
    "Coordenadas del recuadro",
    "矩形座標",
    "边界框坐标"
  ],
  "ui.positive_negative_points": [
    "Positive / negative points",
    "Puntos positivos / negativos",
    "正例 / 負例の点",
    "正点 / 负点"
  ],
  "ui.box_x0": [
    "Box x0",
    "Recuadro x0",
    "矩形 x0",
    "边界框 x0"
  ],
  "ui.box_y0": [
    "Box y0",
    "Recuadro y0",
    "矩形 y0",
    "边界框 y0"
  ],
  "ui.box_x1": [
    "Box x1",
    "Recuadro x1",
    "矩形 x1",
    "边界框 x1"
  ],
  "ui.box_y1": [
    "Box y1",
    "Recuadro y1",
    "矩形 y1",
    "边界框 y1"
  ],
  "ui.points_json": [
    "Points JSON, e.g. [[60.5,80],[100,120]]",
    "Puntos JSON, p. ej. [[60.5,80],[100,120]]",
    "点の JSON。例: [[60.5,80],[100,120]]",
    "点坐标 JSON，例如 [[60.5,80],[100,120]]"
  ],
  "ui.point_labels_json": [
    "Point labels JSON, e.g. [1,0]",
    "Etiquetas de puntos JSON, p. ej. [1,0]",
    "点ラベルの JSON。例: [1,0]",
    "点标签 JSON，例如 [1,0]"
  ],
  "ui.segmentation_alternatives": [
    "Segmentation alternatives",
    "Alternativas de segmentación",
    "領域分割の候補",
    "分割候选"
  ],
  "ui.single_mask": [
    "Single mask",
    "Una máscara",
    "単一マスク",
    "单个蒙版"
  ],
  "ui.native_multimask": [
    "Native multimask alternatives",
    "Alternativas multimáscara nativas",
    "ネイティブ複数マスク候補",
    "原生多蒙版候选"
  ],
  "ui.mask_threshold": [
    "Mask threshold",
    "Umbral de máscara",
    "マスクのしきい値",
    "蒙版阈值"
  ],
  "ui.audio_action": [
    "Audio action",
    "Acción de audio",
    "音声処理",
    "音频操作"
  ],
  "ui.transcribe": [
    "Transcribe",
    "Transcribir",
    "書き起こす",
    "转录"
  ],
  "ui.translate_english": [
    "Translate speech to English",
    "Traducir voz al inglés",
    "音声を英語に翻訳",
    "将语音翻译为英语"
  ],
  "ui.translation_model": [
    "Translation model (optional composed ASR + NMT)",
    "Modelo de traducción (ASR + NMT combinado opcional)",
    "翻訳モデル（ASR + NMT の組み合わせ、任意）",
    "翻译模型（可选 ASR + NMT 组合）"
  ],
  "ui.nmt_source": [
    "NMT source language ID (required with translation model)",
    "ID del idioma origen NMT (obligatorio con modelo de traducción)",
    "NMT 翻訳元言語 ID（翻訳モデル使用時は必須）",
    "NMT 源语言 ID（使用翻译模型时必填）"
  ],
  "ui.nmt_target": [
    "English NMT target language ID (optional)",
    "ID NMT de inglés de destino (opcional)",
    "英語の NMT 翻訳先言語 ID（任意）",
    "英语 NMT 目标语言 ID（可选）"
  ],
  "ui.nmt_target_placeholder": [
    "Translation model default, then eng_Latn; e.g. en",
    "Predeterminado del modelo; luego eng_Latn; p. ej. en",
    "翻訳モデルの既定値、なければ eng_Latn。例: en",
    "优先使用翻译模型默认值，否则使用 eng_Latn；例如 en"
  ],
  "ui.premise": [
    "Premise / source text",
    "Premisa / texto de origen",
    "前提 / 元の文章",
    "前提 / 源文本"
  ],
  "ui.premise_placeholder": [
    "The source text to compare",
    "Texto de origen que comparar",
    "比較する元の文章",
    "用于比较的源文本"
  ],
  "ui.hypothesis": [
    "Hypothesis / statement",
    "Hipótesis / afirmación",
    "仮説 / 主張",
    "假设 / 陈述"
  ],
  "ui.hypothesis_placeholder": [
    "The statement to classify against that source",
    "Afirmación que clasificar respecto al origen",
    "元の文章に対して分類する主張",
    "相对于源文本进行分类的陈述"
  ],
  "ui.pair_hint": [
    "Returns the model’s native labels. A prediction does not verify a fact.",
    "Devuelve las etiquetas nativas del modelo. Una predicción no verifica un hecho.",
    "モデル本来のラベルを返します。予測は事実の検証ではありません。",
    "返回模型的原生标签。预测不等于事实验证。"
  ],
  "ui.input_file": [
    "Input file",
    "Archivo de entrada",
    "入力ファイル",
    "输入文件"
  ],
  "ui.input_mask": [
    "Mask (optional RGBA PNG)",
    "Máscara (PNG RGBA opcional)",
    "マスク（任意の RGBA PNG）",
    "蒙版（可选 RGBA PNG）"
  ],
  "ui.mask_help": [
    "Transparent pixels mark the area to edit; opaque pixels mark the area to keep. Use only fully transparent or fully opaque alpha, with exactly the source image’s dimensions. Images and masks are not resized for masked edits.",
    "Los píxeles transparentes indican dónde editar y los opacos dónde conservar. Usa alfa totalmente transparente u opaco y las mismas dimensiones que la imagen. No se redimensionan imágenes ni máscaras en estas ediciones.",
    "透明な画素は編集領域、不透明な画素は保持領域です。アルファは完全透明か完全不透明にし、元画像と同じサイズにしてください。マスク編集では画像やマスクを拡大縮小しません。",
    "透明像素表示编辑区域，不透明像素表示保留区域。仅使用完全透明或完全不透明的 alpha，并确保尺寸与源图像完全一致。蒙版编辑不会缩放图像或蒙版。"
  ],
  "ui.preserve_pixels": [
    "Keep pixels outside mask",
    "Conservar píxeles fuera de la máscara",
    "マスク外の画素を保持",
    "保留蒙版外的像素"
  ],
  "ui.preserve_on": [
    "On — copy the original RGB pixels after generation",
    "Sí — copiar los píxeles RGB originales tras generar",
    "オン — 生成後に元の RGB 画素をコピー",
    "开启 — 生成后复制原始 RGB 像素"
  ],
  "ui.preserve_off": [
    "Off — return the model’s output without copying",
    "No — devolver la salida sin copiar píxeles",
    "オフ — コピーせずモデルの出力を返す",
    "关闭 — 直接返回模型输出，不复制像素"
  ],
  "ui.vad_check": [
    "Speech activity check",
    "Comprobación de actividad de voz",
    "音声活動の確認",
    "语音活动检查"
  ],
  "ui.vad_hint": [
    "A configured check can skip transcription when no speech is detected.",
    "La comprobación puede omitir la transcripción si no detecta voz.",
    "設定済みの確認処理は、音声が検出されない場合に書き起こしを省略できます。",
    "配置检查后，可在未检测到语音时跳过转录。"
  ],
  "ui.max_tokens": [
    "Max tokens",
    "Máximo de tokens",
    "最大トークン数",
    "最大词元数"
  ],
  "ui.temperature": [
    "Temperature",
    "Temperatura",
    "温度",
    "温度"
  ],
  "ui.top_p": [
    "Top p",
    "Top p",
    "Top p",
    "Top p"
  ],
  "ui.top_k": [
    "Top k",
    "Top k",
    "Top k",
    "Top k"
  ],
  "ui.min_p": [
    "Min p",
    "Min p",
    "Min p",
    "Min p"
  ],
  "ui.repeat_penalty": [
    "Repeat penalty",
    "Penalización por repetición",
    "反復ペナルティ",
    "重复惩罚"
  ],
  "ui.voice": [
    "Voice",
    "Voz",
    "音声",
    "音色"
  ],
  "ui.input_tokens": [
    "Input token limit",
    "Límite de tokens de entrada",
    "入力トークン上限",
    "输入词元上限"
  ],
  "ui.pair_truncation": [
    "Pair input truncation",
    "Recorte de entrada de pares",
    "文章ペア入力の切り詰め",
    "文本对输入截断"
  ],
  "ui.reject_pairs": [
    "Reject oversized pairs",
    "Rechazar pares demasiado grandes",
    "上限超過の文章ペアを拒否",
    "拒绝过大的文本对"
  ],
  "ui.truncate_premise": [
    "Allow premise-only truncation",
    "Permitir recorte solo de la premisa",
    "前提のみの切り詰めを許可",
    "仅允许截断前提文本"
  ],
  "ui.min_sentiment": [
    "Minimum sentiment score",
    "Puntuación mínima de sentimiento",
    "感情スコアの下限",
    "最低情感分数"
  ],
  "ui.min_margin": [
    "Minimum score margin",
    "Margen mínimo de puntuación",
    "スコア差の下限",
    "最小分数差"
  ],
  "ui.detection_threshold": [
    "Detection threshold",
    "Umbral de detección",
    "検出しきい値",
    "检测阈值"
  ],
  "ui.nms": [
    "Filter overlapping boxes",
    "Filtrar recuadros superpuestos",
    "重なる矩形をフィルター",
    "筛选重叠边界框"
  ],
  "ui.model_server_default": [
    "Use model/server default",
    "Usar el valor del modelo/servidor",
    "モデル/サーバーの既定値を使用",
    "使用模型/服务器默认值"
  ],
  "ui.nms_iou": [
    "Overlap IoU threshold",
    "Umbral IoU de solapamiento",
    "重なりの IoU しきい値",
    "重叠 IoU 阈值"
  ],
  "ui.text_guidance": [
    "Text guidance",
    "Guía de texto",
    "テキストガイダンス",
    "文本引导"
  ],
  "ui.image_guidance": [
    "Image guidance",
    "Guía de imagen",
    "画像ガイダンス",
    "图像引导"
  ],
  "ui.image_count": [
    "Image count",
    "Cantidad de imágenes",
    "画像数",
    "图像数量"
  ],
  "ui.ocr_language": [
    "OCR language",
    "Idioma OCR",
    "OCR 言語",
    "OCR 语言"
  ],
  "ui.ocr_language_placeholder": [
    "Model / server default, e.g. eng",
    "Valor del modelo / servidor, p. ej. eng",
    "モデル / サーバーの既定値。例: eng",
    "模型 / 服务器默认值，例如 eng"
  ],
  "ui.ocr_page": [
    "OCR page mode",
    "Modo de página OCR",
    "OCR ページモード",
    "OCR 页面模式"
  ],
  "ui.page_auto": [
    "Automatic page",
    "Página automática",
    "自動ページ",
    "自动页面"
  ],
  "ui.single_column": [
    "Single column",
    "Una columna",
    "単一列",
    "单列"
  ],
  "ui.vertical_block": [
    "Vertical text block",
    "Bloque de texto vertical",
    "縦書きテキストブロック",
    "竖排文本块"
  ],
  "ui.single_block": [
    "Single text block",
    "Un bloque de texto",
    "単一テキストブロック",
    "单个文本块"
  ],
  "ui.single_line": [
    "Single line",
    "Una línea",
    "単一行",
    "单行"
  ],
  "ui.single_word": [
    "Single word",
    "Una palabra",
    "単語一つ",
    "单个词"
  ],
  "ui.circle_word": [
    "Word in a circle",
    "Palabra en un círculo",
    "円内の単語",
    "圆形区域中的词"
  ],
  "ui.single_character": [
    "Single character",
    "Un carácter",
    "文字一つ",
    "单个字符"
  ],
  "ui.sparse_text": [
    "Sparse text",
    "Texto disperso",
    "散在テキスト",
    "稀疏文本"
  ],
  "ui.raw_line": [
    "Raw line",
    "Línea sin procesar",
    "生の行",
    "原始行"
  ],
  "ui.ocr_geometry": [
    "OCR geometry",
    "Geometría OCR",
    "OCR 位置情報",
    "OCR 几何信息"
  ],
  "ui.word_boxes": [
    "Include word boxes and native confidence",
    "Incluir recuadros de palabras y confianza nativa",
    "単語の矩形とネイティブ信頼度を含める",
    "包含词边界框和原生置信度"
  ],
  "ui.no_geometry": [
    "Text without geometry",
    "Texto sin geometría",
    "位置情報なしのテキスト",
    "不含几何信息的文本"
  ],
  "ui.image_size": [
    "Image size",
    "Tamaño de imagen",
    "画像サイズ",
    "图像尺寸"
  ],
  "ui.steps": [
    "Steps",
    "Pasos",
    "ステップ数",
    "步数"
  ],
  "ui.seed": [
    "Seed",
    "Semilla",
    "シード",
    "随机种子"
  ],
  "ui.video_frames": [
    "Video frames",
    "Fotogramas de vídeo",
    "動画フレーム数",
    "视频帧数"
  ],
  "ui.fps": [
    "FPS",
    "FPS",
    "FPS",
    "帧率"
  ],
  "ui.target_language": [
    "Target language",
    "Idioma de destino",
    "翻訳先言語",
    "目标语言"
  ],
  "ui.for_translation": [
    "For translation",
    "Para traducción",
    "翻訳用",
    "用于翻译"
  ],
  "ui.key_placeholder": [
    "Only if configured",
    "Solo si está configurada",
    "設定されている場合のみ",
    "仅在已配置时需要"
  ],
  "ui.variant_help": [
    "Enter fp16 when the checkpoint requires that weight suffix. Leave empty for ordinary filenames; this does not change compute precision.",
    "Introduce fp16 si el modelo requiere ese sufijo de pesos. Déjalo vacío para nombres normales; no cambia la precisión del cálculo.",
    "重みのファイル名に fp16 接尾辞が必要な場合に指定します。通常は空欄にしてください。計算精度は変更されません。",
    "当权重文件需要该后缀时输入 fp16。普通文件名请留空；此项不会改变计算精度。"
  ],
  "ui.initial_checkpoint_help": [
    "Point to a complete local checkpoint. Model weights stay outside Titanos.",
    "Selecciona un modelo local completo. Los pesos permanecen fuera de Titanos.",
    "完全なローカルモデルを指定してください。重みは Titanos の外部に保持されます。",
    "请选择完整的本地模型。权重始终保存在 Titanos 外部。"
  ],
  "task.tensor": [
    "Tensor (typed API)",
    "Tensor (API tipada)",
    "テンソル（型付き API）",
    "张量（带类型 API）"
  ],
  "ui.server_status": [
    "Server returned {{status}}",
    "El servidor devolvió {{status}}",
    "サーバーが {{status}} を返しました",
    "服务器返回 {{status}}"
  ],
  "ui.invalid_value": [
    "Invalid value",
    "Valor no válido",
    "無効な値",
    "无效值"
  ],
  "error.authentication": [
    "Check the server API key under Advanced, Connection.",
    "Comprueba la clave API del servidor en Avanzado, Conexión.",
    "「詳細」の「接続」でサーバー API キーを確認してください。",
    "请在“高级”的“连接”中检查服务器 API 密钥。"
  ],
  "error.permission": [
    "This operation is not allowed by the server configuration.",
    "La configuración del servidor no permite esta operación.",
    "サーバー設定により、この操作は許可されていません。",
    "服务器配置不允许此操作。"
  ],
  "error.device_unavailable": [
    "Restore the selected device or choose an available one before loading.",
    "Restablece el dispositivo elegido o selecciona uno disponible antes de cargar.",
    "ロード前に選択したデバイスを復旧するか、利用可能なものを選んでください。",
    "加载前请恢复所选设备或选择可用设备。"
  ],
  "error.runtime_missing": [
    "Prepare a compatible runtime, then retry.",
    "Prepara un entorno compatible y vuelve a intentarlo.",
    "互換性のある実行環境を準備して再試行してください。",
    "请准备兼容的运行时后重试。"
  ],
  "error.acquisition_not_prepared": [
    "Wait for acquisition to finish before activating the model.",
    "Espera a que termine la adquisición antes de activar el modelo.",
    "モデルを有効化する前に、取得の完了を待ってください。",
    "请等待获取完成后再激活模型。"
  ],
  "error.provider_error": [
    "Check provider access and connectivity, then retry.",
    "Comprueba el acceso al proveedor y la conexión; vuelve a intentarlo.",
    "提供元へのアクセスと接続を確認して再試行してください。",
    "请检查提供方访问权限和连接后重试。"
  ],
  "error.invalid_backend_response": [
    "The model returned an invalid result. Check runtime details before retrying.",
    "El modelo devolvió un resultado no válido. Revisa el entorno antes de reintentar.",
    "モデルの結果が無効です。実行環境の詳細を確認して再試行してください。",
    "模型返回了无效结果。请检查运行时详情后重试。"
  ],
  "ui.choose_source": [
    "Choose a source image for this instruction edit",
    "Elige una imagen de origen para esta edición por instrucciones",
    "この指示編集に使う元画像を選んでください",
    "请为此指令编辑选择源图像"
  ],
  "ui.mask_needs_source": [
    "Choose a source image before adding a mask",
    "Elige una imagen antes de añadir una máscara",
    "マスクを追加する前に元画像を選んでください",
    "添加蒙版前请先选择源图像"
  ],
  "ui.choose_preservation": [
    "Choose whether to keep pixels outside the mask",
    "Elige si quieres conservar los píxeles fuera de la máscara",
    "マスク外の画素を保持するか選んでください",
    "请选择是否保留蒙版外的像素"
  ],
  "ui.enter_query": [
    "Enter a reranking query",
    "Introduce una consulta de reordenación",
    "再順位付けのクエリを入力してください",
    "请输入重排查询"
  ],
  "ui.enter_documents": [
    "Enter at least one nonempty document line",
    "Introduce al menos una línea de documento no vacía",
    "空でない文書を最低 1 行入力してください",
    "请输入至少一行非空文档"
  ],
  "ui.choose_rerank_truncation": [
    "Choose a supported rerank truncation policy",
    "Elige una política de recorte compatible",
    "対応する再順位付けの切り詰め方針を選んでください",
    "请选择受支持的重排截断策略"
  ],
  "ui.choose_return_documents": [
    "Choose whether rerank results include documents",
    "Elige si los resultados incluyen los documentos",
    "再順位付けの結果に文書を含めるか選んでください",
    "请选择重排结果是否包含文档"
  ],
  "ui.choose_audio": [
    "Choose an audio file",
    "Elige un archivo de audio",
    "音声ファイルを選んでください",
    "请选择音频文件"
  ],
  "ui.choose_vad_probabilities": [
    "Choose whether VAD returns frame probabilities",
    "Elige si VAD devuelve probabilidades por fotograma",
    "VAD がフレーム確率を返すか選んでください",
    "请选择 VAD 是否返回帧概率"
  ],
  "ui.enter_pair": [
    "Enter both the premise and hypothesis",
    "Introduce la premisa y la hipótesis",
    "前提と仮説の両方を入力してください",
    "请同时输入前提和假设"
  ],
  "ui.positive_token_limit": [
    "Input token limit must be a positive safe integer",
    "El límite de tokens debe ser un entero positivo seguro",
    "入力トークン上限は安全な正の整数にしてください",
    "输入词元上限必须是正的安全整数"
  ],
  "ui.choose_pair_truncation": [
    "Choose a supported pair truncation policy",
    "Elige una política compatible de recorte de pares",
    "対応する文章ペアの切り詰め方針を選んでください",
    "请选择受支持的文本对截断策略"
  ],
  "ui.choose_image": [
    "Choose an image",
    "Elige una imagen",
    "画像を選んでください",
    "请选择图像"
  ],
  "ui.ocr_language_invalid": [
    "Use a Tesseract language code such as eng or eng+deu",
    "Usa un código Tesseract como eng o eng+deu",
    "eng や eng+deu などの Tesseract 言語コードを使用してください",
    "请使用 eng 或 eng+deu 等 Tesseract 语言代码"
  ],
  "ui.choose_ocr_page": [
    "Choose a supported OCR page mode",
    "Elige un modo de página OCR compatible",
    "対応する OCR ページモードを選んでください",
    "请选择受支持的 OCR 页面模式"
  ],
  "ui.choose_ocr_geometry": [
    "Choose a supported OCR geometry option",
    "Elige una opción de geometría OCR compatible",
    "対応する OCR 位置情報オプションを選んでください",
    "请选择受支持的 OCR 几何选项"
  ],
  "ui.choose_nmt_source": [
    "Choose the NMT source language ID for the translation model",
    "Elige el ID del idioma origen NMT del modelo de traducción",
    "翻訳モデルの NMT 翻訳元言語 ID を選んでください",
    "请选择翻译模型的 NMT 源语言 ID"
  ],
  "ui.enter_box": [
    "Enter all four box coordinates",
    "Introduce las cuatro coordenadas del recuadro",
    "矩形の 4 座標をすべて入力してください",
    "请输入全部四个边界框坐标"
  ],
  "ui.box_finite": [
    "Box coordinates must be finite numbers",
    "Las coordenadas deben ser números finitos",
    "矩形座標は有限数にしてください",
    "边界框坐标必须是有限数值"
  ],
  "ui.box_order": [
    "Segmentation box must be nonnegative and ordered with x0 < x1 and y0 < y1",
    "El recuadro debe ser no negativo y cumplir x0 < x1 e y0 < y1",
    "矩形座標は非負で x0 < x1、y0 < y1 にしてください",
    "分割边界框必须为非负数，并满足 x0 < x1 且 y0 < y1"
  ],
  "ui.points_valid_json": [
    "Points and point labels must be valid JSON arrays",
    "Los puntos y etiquetas deben ser matrices JSON válidas",
    "点と点ラベルは有効な JSON 配列にしてください",
    "点坐标和点标签必须是有效的 JSON 数组"
  ],
  "ui.choose_segmentation": [
    "Choose a segmentation prompt type",
    "Elige un tipo de indicación de segmentación",
    "領域分割の指示形式を選んでください",
    "请选择分割提示类型"
  ],
  "ui.choose_multimask": [
    "Choose a supported segmentation alternatives option",
    "Elige una opción compatible de alternativas de segmentación",
    "対応する領域分割候補の設定を選んでください",
    "请选择受支持的分割候选选项"
  ],
  "ui.choose_backend_task": [
    "Choose a task supported by the selected backend",
    "Elige una tarea compatible con el motor seleccionado",
    "選択したバックエンドが対応するタスクを選んでください",
    "请选择所选后端支持的任务"
  ],
  "ui.voice_map_invalid": [
    "Voice map must be a JSON object",
    "El mapa de voces debe ser un objeto JSON",
    "音声マップは JSON オブジェクトにしてください",
    "音色映射必须是 JSON 对象"
  ],
  "ui.voice_paths_absolute": [
    "Voice map values must be absolute local WAV paths",
    "Las voces deben usar rutas locales absolutas a archivos WAV",
    "音声マップの値はローカル WAV の絶対パスにしてください",
    "音色映射值必须是本地 WAV 文件的绝对路径"
  ],
  "ui.variant_invalid": [
    "Weight variant must contain only ASCII letters, numbers, underscores or hyphens, such as fp16 or non_ema; leave empty for ordinary filenames",
    "La variante solo admite letras ASCII, números, guiones bajos o guiones, como fp16 o non_ema; déjala vacía para nombres normales",
    "重みの種類は fp16 や non_ema のように ASCII 英数字、下線、ハイフンのみです。通常のファイル名は空欄にしてください",
    "权重变体只能包含 ASCII 字母、数字、下划线或连字符，例如 fp16 或 non_ema；普通文件名请留空"
  ],
  "ui.choose_real_dml": [
    "Choose a current non-software DirectML adapter",
    "Elige un adaptador DirectML actual que no sea de software",
    "現在の非ソフトウェア DirectML アダプターを選んでください",
    "请选择当前的非软件 DirectML 适配器"
  ],
  "ui.page_title": [
    "Titanos · Local inference",
    "Titanos · Inferencia local",
    "Titanos · ローカル推論",
    "Titanos · 本地推理"
  ],
  "ui.instruction": [
    "Instruction",
    "Instrucción",
    "編集指示",
    "编辑指令"
  ],
  "ui.query": [
    "Query",
    "Consulta",
    "クエリ",
    "查询"
  ],
  "ui.load": [
    "Load",
    "Cargar",
    "ロード",
    "加载"
  ],
  "ui.unload": [
    "Unload",
    "Liberar",
    "アンロード",
    "卸载"
  ],
  "ui.show_runtime_for": [
    "Show runtime details for",
    "Mostrar detalles del entorno de",
    "実行環境の詳細を表示:",
    "显示运行时详情："
  ],
  "ui.remove_registration_for": [
    "Remove registration for",
    "Eliminar registro de",
    "登録を削除:",
    "删除注册："
  ],
  "ui.lifecycle_aria": [
    "{{action}} {{model}}",
    "{{action}} {{model}}",
    "{{action}} {{model}}",
    "{{action}} {{model}}"
  ],
  "ui.image_size_example": [
    "e.g. 1024x1024",
    "p. ej. 1024x1024",
    "例: 1024x1024",
    "例如 1024x1024"
  ],
  "ui.image_size_limit": [
    "WIDTHxHEIGHT · up to {{dimension}}px per side{{pixels}}",
    "ANCHOxALTO · hasta {{dimension}} px por lado{{pixels}}",
    "幅x高さ · 各辺最大 {{dimension}} px{{pixels}}",
    "宽x高 · 每边最多 {{dimension}} 像素{{pixels}}"
  ],
  "ui.pixel_limit": [
    " and {{count}} pixels",
    " y {{count}} píxeles",
    "、総画素数 {{count}} 以下",
    "，总计不超过 {{count}} 像素"
  ],
  "ui.model_missing_controls": [
    "Incomplete model capability response",
    "Respuesta de capacidades incompleta",
    "モデル機能の応答が不完全です",
    "模型功能响应不完整"
  ],
  "ui.unsupported_instruction": [
    "Unsupported instruction-edit capability profile",
    "Perfil de edición por instrucciones no compatible",
    "非対応の指示編集プロファイルです",
    "不支持此指令编辑功能配置"
  ],
  "ui.invalid_catalog": [
    "Invalid model catalog",
    "Catálogo de modelos no válido",
    "モデルカタログが無効です",
    "模型目录无效"
  ],
  "ui.unexpected_asset": [
    "Unexpected asset location",
    "Ubicación inesperada del recurso",
    "予期しないアセット保存先です",
    "意外的资源位置"
  ],
  "ui.depth_invalid": [
    "Invalid relative-depth result metadata",
    "Metadatos de profundidad relativa no válidos",
    "相対深度の結果メタデータが無効です",
    "相对深度结果元数据无效"
  ],
  "ui.depth_base64_invalid": [
    "Invalid relative-depth base64 data",
    "Datos base64 de profundidad relativa no válidos",
    "相対深度の base64 データが無効です",
    "相对深度 base64 数据无效"
  ],
  "ui.depth_bytes_invalid": [
    "Relative-depth byte count does not match its dimensions",
    "Los bytes de profundidad no coinciden con las dimensiones",
    "相対深度のバイト数が画像サイズと一致しません",
    "相对深度字节数与尺寸不匹配"
  ],
  "ui.depth_finite": [
    "Relative-depth values must be finite",
    "Los valores de profundidad deben ser finitos",
    "相対深度値は有限数である必要があります",
    "相对深度值必须为有限数值"
  ],
  "ui.unsupported_task": [
    "Unknown or unsupported Studio task: {{task}}",
    "Tarea de Studio desconocida o no compatible: {{task}}",
    "不明または非対応の Studio タスク: {{task}}",
    "未知或不受支持的 Studio 任务：{{task}}"
  ],
  "ui.no_request_adapter": [
    "Studio has no request adapter for {{task}}",
    "Studio no tiene un adaptador de solicitudes para {{task}}",
    "Studio に {{task}} のリクエストアダプターがありません",
    "Studio 没有 {{task}} 请求适配器"
  ],
  "ui.vector_preview": [
    "{{count}} vectors · {{dimensions}} dimensions\nPreview: {{preview}}",
    "{{count}} vectores · {{dimensions}} dimensiones\nVista previa: {{preview}}",
    "{{count}} ベクトル · {{dimensions}} 次元\nプレビュー: {{preview}}",
    "{{count}} 个向量 · {{dimensions}} 维\n预览：{{preview}}"
  ],
  "ui.detection_summary": [
    "{{count}} detected objects{{size}}. Native scores are not verified truth.",
    "{{count}} objetos detectados{{size}}. Las puntuaciones nativas no son hechos verificados.",
    "検出物体 {{count}} 個{{size}}。ネイティブスコアは検証済みの事実ではありません。",
    "检测到 {{count}} 个对象{{size}}。原生分数不是已验证的事实。"
  ],
  "ui.input_size": [
    " · {{width}}×{{height}} input",
    " · entrada {{width}}×{{height}}",
    " · 入力 {{width}}×{{height}}",
    " · 输入 {{width}}×{{height}}"
  ],
  "ui.detection_item": [
    "{{label}} · {{score}} · box {{box}}",
    "{{label}} · {{score}} · recuadro {{box}}",
    "{{label}} · {{score}} · 矩形 {{box}}",
    "{{label}} · {{score}} · 边界框 {{box}}"
  ],
  "ui.rerank_item": [
    "Document {{index}} · raw score {{score}}{{text}}",
    "Documento {{index}} · puntuación bruta {{score}}{{text}}",
    "文書 {{index}} · 生スコア {{score}}{{text}}",
    "文档 {{index}} · 原始分数 {{score}}{{text}}"
  ],
  "ui.vad_result": [
    "{{count}} speech intervals · {{kind}} · uncalibrated.",
    "{{count}} intervalos de voz · {{kind}} · sin calibrar.",
    "音声区間 {{count}} 件 · {{kind}} · 未校正。",
    "{{count}} 个语音区间 · {{kind}} · 未经校准。"
  ],
  "ui.vad_interval": [
    "{{start}}–{{end}} seconds · samples {{first}}–{{last}}",
    "{{start}}–{{end}} segundos · muestras {{first}}–{{last}}",
    "{{start}}–{{end}} 秒 · サンプル {{first}}–{{last}}",
    "{{start}}–{{end}} 秒 · 样本 {{first}}–{{last}}"
  ],
  "ui.mask_item": [
    "Mask {{index}} · {{pixels}} selected pixels · predicted IoU {{iou}} · uncalibrated",
    "Máscara {{index}} · {{pixels}} píxeles seleccionados · IoU previsto {{iou}} · sin calibrar",
    "マスク {{index}} · 選択画素 {{pixels}} · 予測 IoU {{iou}} · 未校正",
    "蒙版 {{index}} · 已选择 {{pixels}} 个像素 · 预测 IoU {{iou}} · 未经校准"
  ],
  "ui.mask_alt": [
    "Segmentation mask {{index}}; transparent pixels are selected for editing",
    "Máscara {{index}}; los píxeles transparentes se editarán",
    "領域分割マスク {{index}}。透明な画素が編集対象",
    "分割蒙版 {{index}}；透明像素被选中进行编辑"
  ],
  "ui.pair_prediction": [
    "Pair {{index}} · {{id}} · Predicted native label: {{label}} (class {{class}})",
    "Par {{index}} · {{id}} · Etiqueta nativa prevista: {{label}} (clase {{class}})",
    "ペア {{index}} · {{id}} · 予測ネイティブラベル: {{label}}（クラス {{class}}）",
    "文本对 {{index}} · {{id}} · 预测原生标签：{{label}}（类别 {{class}}）"
  ],
  "ui.pair_accounting": [
    "Input tokens: {{input}}; processed: {{processed}}; premise truncated: {{truncated}}. Hypothesis preserved.",
    "Tokens de entrada: {{input}}; procesados: {{processed}}; premisa recortada: {{truncated}}. Hipótesis conservada.",
    "入力トークン: {{input}}、処理済み: {{processed}}、前提の切り詰め: {{truncated}}。仮説は保持されています。",
    "输入词元：{{input}}；已处理：{{processed}}；前提已截断：{{truncated}}。假设文本保持不变。"
  ],
  "ui.point_rules": [
    "Points require nonnegative finite [x,y] pairs and one 0/1 label per point{{limit}}",
    "Los puntos requieren pares [x,y] finitos y no negativos y una etiqueta 0/1 por punto{{limit}}",
    "点は非負の有限な [x,y] ペアと、各点に 0/1 ラベルが必要です{{limit}}",
    "点必须是非负有限 [x,y] 坐标对，且每点对应一个 0/1 标签{{limit}}"
  ],
  "ui.point_limit": [
    "; at most {{count}} points are configured",
    "; se permiten como máximo {{count}} puntos",
    "。設定上限は {{count}} 点です",
    "；配置上限为 {{count}} 个点"
  ],
  "ui.provider_ace": [
    "ACE-Step 1.5 · music",
    "ACE-Step 1.5 · música",
    "ACE-Step 1.5 · 音楽",
    "ACE-Step 1.5 · 音乐"
  ],
  "ui.unnamed": [
    "Unnamed",
    "Sin nombre",
    "名称なし",
    "未命名"
  ],
  "ui.byte_count": [
    "{{count}} bytes",
    "{{count}} bytes",
    "{{count}} バイト",
    "{{count}} 字节"
  ],
  "retrieval.title": [
    "Answer from your documents",
    "Respuestas desde tus documentos",
    "文書に基づく回答",
    "根据你的文档回答"
  ],
  "retrieval.intro": [
    "Choose three registered models and a JSON array of passages with unique id and text fields. Files are sent to this local Titanos server.",
    "Elige tres modelos registrados y una matriz JSON de pasajes con campos id únicos y text. Los archivos se envían a este servidor local de Titanos.",
    "登録済みの 3 モデルと、一意な id と text を持つ文章の JSON 配列を選んでください。ファイルはこのローカル Titanos サーバーへ送信されます。",
    "请选择三个已注册模型，以及包含唯一 id 和 text 字段的段落 JSON 数组。文件会发送到此本地 Titanos 服务器。"
  ],
  "retrieval.embedding": [
    "Embedding model",
    "Modelo de embeddings",
    "埋め込みモデル",
    "嵌入模型"
  ],
  "retrieval.reranker": [
    "Reranker",
    "Modelo de reordenación",
    "再順位付けモデル",
    "重排模型"
  ],
  "retrieval.answer_model": [
    "Answer model",
    "Modelo de respuesta",
    "回答モデル",
    "回答模型"
  ],
  "retrieval.refresh": [
    "Refresh workflow models",
    "Actualizar modelos del flujo",
    "ワークフローのモデルを更新",
    "刷新工作流模型"
  ],
  "retrieval.passages_file": [
    "Passages JSON file",
    "Archivo JSON de pasajes",
    "文章の JSON ファイル",
    "段落 JSON 文件"
  ],
  "retrieval.question": [
    "Question",
    "Pregunta",
    "質問",
    "问题"
  ],
  "retrieval.settings": [
    "Retrieval and answer settings",
    "Ajustes de búsqueda y respuesta",
    "検索と回答の設定",
    "检索和回答设置"
  ],
  "retrieval.candidates": [
    "Candidates to rerank",
    "Candidatos para reordenar",
    "再順位付けする候補数",
    "待重排候选数量"
  ],
  "retrieval.context_passages": [
    "Passages supplied to answer model",
    "Pasajes enviados al modelo de respuesta",
    "回答モデルへ渡す文章数",
    "提供给回答模型的段落数量"
  ],
  "retrieval.encoder_tokens": [
    "Encoder token limit",
    "Límite de tokens del codificador",
    "エンコーダーのトークン上限",
    "编码器词元上限"
  ],
  "retrieval.rerank_tokens": [
    "Reranker token limit",
    "Límite de tokens del reordenador",
    "再順位付けモデルのトークン上限",
    "重排模型词元上限"
  ],
  "retrieval.input_tokens": [
    "Answer input token limit",
    "Límite de tokens de entrada para la respuesta",
    "回答入力のトークン上限",
    "回答输入词元上限"
  ],
  "retrieval.output_tokens": [
    "Answer output token limit",
    "Límite de tokens de salida de la respuesta",
    "回答出力のトークン上限",
    "回答输出词元上限"
  ],
  "retrieval.deadline": [
    "Whole request deadline (milliseconds)",
    "Plazo total de la solicitud (milisegundos)",
    "リクエスト全体の制限時間（ミリ秒）",
    "整个请求的截止时间（毫秒）"
  ],
  "retrieval.policy": [
    "How to read source passages",
    "Cómo leer los pasajes de origen",
    "出典文章の読み取り方針",
    "源段落读取方式"
  ],
  "retrieval.policy_v2": [
    "Use source facts; ignore embedded commands (v2)",
    "Usar hechos de la fuente; ignorar órdenes insertadas (v2)",
    "出典の事実を使用し、埋め込まれた命令を無視（v2）",
    "使用来源事实；忽略嵌入的命令（v2）"
  ],
  "retrieval.policy_v1": [
    "Original source-reading policy (v1)",
    "Política original de lectura de fuentes (v1)",
    "従来の出典読み取り方針（v1）",
    "原始来源读取策略（v1）"
  ],
  "retrieval.submit": [
    "Answer with sources",
    "Responder con fuentes",
    "出典付きで回答",
    "附来源回答"
  ],
  "retrieval.answer_heading": [
    "Answer with source quotations",
    "Respuesta con citas de las fuentes",
    "出典の引用付き回答",
    "附来源引文的回答"
  ],
  "retrieval.no_answer": [
    "No answer returned",
    "No se devolvió respuesta",
    "回答は返されませんでした",
    "未返回回答"
  ],
  "retrieval.quote_note": [
    "Quotes match submitted text. Review whether they support each claim; answer correctness was not independently verified.",
    "Las citas coinciden con el texto enviado. Revisa si respaldan cada afirmación; la respuesta no se verificó de forma independiente.",
    "引用は送信した文章と一致しています。各主張を裏付けるか確認してください。回答の正しさは独立に検証していません。",
    "引文与提交的文本一致。请检查是否支持各项陈述；回答正确性尚未经过独立验证。"
  ],
  "retrieval.source_ref": [
    "Caller source reference: {{source}}",
    "Referencia proporcionada: {{source}}",
    "呼び出し元の出典参照: {{source}}",
    "调用方来源引用：{{source}}"
  ],
  "retrieval.quote_positions": [
    "This quote appears at {{count}} positions in the passage.",
    "Esta cita aparece en {{count}} posiciones del pasaje.",
    "この引用は文章内の {{count}} か所に現れます。",
    "此引文出现在段落中的 {{count}} 个位置。"
  ],
  "retrieval.abstained": [
    "The model did not answer from the selected passages. Relevant information may still be present; review the source and stage trace.",
    "El modelo no respondió a partir de los pasajes elegidos. Puede haber información relevante; revisa las fuentes y el registro de etapas.",
    "モデルは選択した文章から回答しませんでした。関連情報が含まれる可能性はあるため、出典と段階別の記録を確認してください。",
    "模型未根据所选段落作答。段落中仍可能包含相关信息，请检查来源和阶段记录。"
  ],
  "retrieval.trace": [
    "Source and stage trace",
    "Fuentes y registro de etapas",
    "出典と段階別の記録",
    "来源和阶段记录"
  ],
  "retrieval.missing_models": [
    "Register a compatible encoder, reranker and native answer model to use this workflow.",
    "Registra un codificador, un reordenador y un modelo nativo de respuesta compatibles para usar este flujo.",
    "このワークフローには、互換エンコーダー、再順位付けモデル、ネイティブ回答モデルの登録が必要です。",
    "请注册兼容的编码器、重排模型和原生回答模型以使用此工作流。"
  ],
  "retrieval.running": [
    "Retrieving passages and preparing an answer…",
    "Buscando pasajes y preparando una respuesta…",
    "文章を検索し、回答を準備中…",
    "正在检索段落并准备回答…"
  ],
  "retrieval.choose_file": [
    "Choose a passages JSON file",
    "Elige un archivo JSON de pasajes",
    "文章の JSON ファイルを選んでください",
    "请选择段落 JSON 文件"
  ],
  "retrieval.failed": [
    "Request failed; no answer claims returned.",
    "Solicitud fallida; no se devolvieron afirmaciones.",
    "リクエストが失敗し、回答の主張は返されませんでした。",
    "请求失败；未返回任何回答陈述。"
  ],
  "retrieval.finished": [
    "Finished.",
    "Finalizado.",
    "完了しました。",
    "已完成。"
  ],
  "ui.music_runtime_required": [
    "Music runtime required",
    "Se requiere el entorno musical",
    "音楽用実行環境が必要",
    "需要音乐运行时"
  ],
  "ui.music_runtime_missing": [
    "This package cannot run this music model. Use a complete Titanos music distribution with the matching ACE-Step runtime, then inspect the model again. No model download was started.",
    "Este paquete no puede ejecutar el modelo musical. Usa una distribución completa de Titanos con el entorno ACE-Step correspondiente y vuelve a inspeccionar el modelo. No se inició ninguna descarga.",
    "このパッケージでは音楽モデルを実行できません。対応する ACE-Step 実行環境を含む完全な Titanos 音楽配布版を使用し、モデルを再確認してください。モデルのダウンロードは開始していません。",
    "此软件包无法运行该音乐模型。请使用包含匹配 ACE-Step 运行时的完整 Titanos 音乐发行版，然后重新检查模型。尚未开始下载模型。"
  ],
  "ui.music_variant_unsupported": [
    "This music runtime does not support the selected variant. Choose XL SFT or a runtime that explicitly supports this variant before downloading.",
    "Este entorno musical no admite la variante elegida. Selecciona XL SFT o un entorno que admita expresamente esa variante antes de descargar.",
    "この音楽実行環境は選択した種類に対応していません。ダウンロード前に XL SFT、またはその種類を明示的にサポートする実行環境を選んでください。",
    "此音乐运行时不支持所选变体。下载前请选择 XL SFT，或明确支持此变体的运行时。"
  ],
  "brand.family": [
    "Born of HiveMind",
    "Nacido de HiveMind",
    "HiveMind から生まれた",
    "源自 HiveMind"
  ],
  "ui.join": [
    "{{text}}",
    "{{text}}",
    "{{text}}",
    "{{text}}"
  ],
  "ui.recommended_suffix": [
    " · recommended",
    " · recomendado",
    " · 推奨",
    " · 推荐"
  ],
  "ui.provider_notes": [
    "Provider notes (original)",
    "Notas del proveedor (originales)",
    "提供元の注記（原文）",
    "提供方说明（原文）"
  ],
  "ui.capability_error": [
    "Capabilities unavailable: {{error}}",
    "Capacidades no disponibles: {{error}}",
    "機能情報を取得できません: {{error}}",
    "功能信息不可用：{{error}}"
  ],
  "status.unsupported_adapter": [
    "unsupported adapter",
    "adaptador no compatible",
    "アダプター非対応",
    "适配器不支持"
  ],
  "stage.validating_bundle": [
    "validating model bundle",
    "validando el paquete del modelo",
    "モデルバンドルを検証中",
    "正在验证模型包"
  ],
  "stage.bundle_verified": [
    "model bundle verified",
    "paquete del modelo verificado",
    "モデルバンドル検証済み",
    "模型包已验证"
  ],
  "stage.loading_planner": [
    "loading music planner",
    "cargando el planificador musical",
    "音楽プランナーをロード中",
    "正在加载音乐规划器"
  ],
  "stage.loading_dit": [
    "loading diffusion model",
    "cargando el modelo de difusión",
    "拡散モデルをロード中",
    "正在加载扩散模型"
  ],
  "stage.loading_text_encoder": [
    "loading text encoder",
    "cargando el codificador de texto",
    "テキストエンコーダーをロード中",
    "正在加载文本编码器"
  ],
  "stage.loading_vae": [
    "loading audio decoder",
    "cargando el decodificador de audio",
    "音声デコーダーをロード中",
    "正在加载音频解码器"
  ],
  "stage.conditioning_hints": [
    "preparing music hints",
    "preparando las indicaciones musicales",
    "音楽の条件を準備中",
    "正在准备音乐条件提示"
  ],
  "stage.writing_audio": [
    "writing audio",
    "guardando el audio",
    "音声を書き込み中",
    "正在写入音频"
  ],
  "stage.failed": [
    "failed",
    "fallido",
    "失敗",
    "失败"
  ],
  "ui.uncalibrated_suffix": [
    " · uncalibrated",
    " · sin calibrar",
    " · 未校正",
    " · 未经校准"
  ],
  "ui.unchanged_suffix": [
    " · unchanged",
    " · sin cambios",
    " · 変更なし",
    " · 未更改"
  ],
  "ui.translated_suffix": [
    " · translated",
    " · traducido",
    " · 翻訳済み",
    " · 已翻译"
  ]
};
Object.assign(workflowText,{
  'setup.check_blocked':['Setup needs attention','La configuración necesita atención','設定の確認が必要','设置需要检查'],
  'setup.check_unknown':['Compatibility unverified','Compatibilidad sin verificar','互換性は未検証','兼容性尚未验证'],
  'setup.check_ready':['Files ready for a load check','Archivos listos para comprobar la carga','ロード確認用のファイルが揃っています','文件已齐备，可检查加载'],
  'setup.check_verified':['Verified at load','Verificado al cargar','ロード時に検証済み','加载时已验证'],
  'setup.check_observed':['Runtime details of the loaded model confirm the items marked Loaded. Local file checks remain advisory.','Los detalles de ejecución del modelo cargado confirman los elementos marcados como Loaded. La revisión de archivos locales sigue siendo orientativa.','ロード済みモデルの実行時詳細が「Loaded」と表示された項目を確認しています。ローカルファイルの確認は参考情報のままです。','已加载模型的运行时详情确认了标记为 Loaded 的项目。本地文件检查仍仅供参考。'],
  'setup.check_title':['{{model}} · {{state}}','{{model}} · {{state}}','{{model}} · {{state}}','{{model}} · {{state}}'],
  'setup.check_scope':['Local file checks are advisory. Loading verifies runtime compatibility.','La revisión de archivos locales es orientativa. La carga verifica la compatibilidad del entorno.','ローカルファイルの確認は参考情報です。実行環境との互換性はロード時に検証されます。','本地文件检查仅供参考。加载时才会验证运行时兼容性。'],
  'setup.check_files':['Files: {{files}}','Archivos: {{files}}','ファイル：{{files}}','文件：{{files}}'],
  'setup.check_next':['Next: {{action}}','Siguiente: {{action}}','次の手順：{{action}}','下一步：{{action}}'],
  'task.diarization':['Find speakers and overlap','Detectar hablantes y solapamientos','話者と重複発話を検出','检测说话人与重叠语音'],
  'audio.analyze':['Transcribe with speakers and overlap','Transcribir con hablantes y solapamientos','話者と重複発話を含めて文字起こし','转录并标注说话人与重叠语音'],
  'audio.speaker_model':['Speaker model','Modelo de hablantes','話者モデル','说话人模型'],
  'audio.choose_speakers':['Choose a speaker model','Elige un modelo de hablantes','話者モデルを選択','选择说话人模型'],
  'audio.no_text_analysis':['No transcript analysis','Sin análisis de la transcripción','文字起こしを分析しない','不分析转录文本'],
  'audio.transcript_analysis':['Transcript text analysis','Análisis del texto transcrito','文字起こしテキストの分析','转录文本分析'],
  'audio.speaker_count':['Speaker count (0: automatic)','Número de hablantes (0: automático)','話者数（0：自動）','说话人数（0：自动）'],
  'audio.cluster_threshold':['Clustering threshold','Umbral de agrupación','クラスタリングのしきい値','聚类阈值'],
  'audio.minimum_speech':['Minimum speech interval (seconds)','Duración mínima del habla (segundos)','最短発話区間（秒）','最短语音片段（秒）'],
  'audio.minimum_gap':['Fill speech gaps shorter than (seconds)','Unir pausas más cortas que (segundos)','この長さ未満の無音を結合（秒）','合并短于此时长的语音间隔（秒）'],
  'audio.time_budget':['Total time budget (milliseconds)','Tiempo total disponible (milisegundos)','合計処理時間の上限（ミリ秒）','总处理时间上限（毫秒）'],
  'audio.result_summary':['{{count}} anonymous speakers · {{overlaps}} overlapping intervals','{{count}} hablantes anónimos · {{overlaps}} intervalos solapados','匿名話者 {{count}} 人 · 重複発話 {{overlaps}} 区間','{{count}} 位匿名说话人 · {{overlaps}} 个重叠片段'],
  'audio.anonymous_note':['Labels apply only to this recording. Overlap does not separate voices or identify people.','Las etiquetas solo se aplican a esta grabación. El solapamiento no separa voces ni identifica personas.','ラベルはこの録音内だけで有効です。重複発話の検出は音声の分離や人物の特定ではありません。','标签仅适用于本段录音。重叠检测不会分离声音或识别个人身份。'],
  'audio.speaker_interval':['{{speaker}} · {{start}}–{{end}} seconds','{{speaker}} · {{start}}–{{end}} segundos','{{speaker}} · {{start}}～{{end}} 秒','{{speaker}} · {{start}}–{{end}} 秒'],
  'audio.overlap_interval':['Overlap: {{speakers}} · {{start}}–{{end}} seconds','Solapamiento: {{speakers}} · {{start}}–{{end}} segundos','重複発話：{{speakers}} · {{start}}～{{end}} 秒','重叠：{{speakers}} · {{start}}–{{end}} 秒'],
  'audio.overlap':['Overlap','Solapamiento','重複発話','重叠'],
  'audio.unattributed':['Unattributed','Sin atribuir','話者未割り当て','未归属'],
  'audio.speaker_text':['{{speakers}}: {{text}}','{{speakers}}: {{text}}','{{speakers}}：{{text}}','{{speakers}}：{{text}}'],
  'audio.outside_timing':['The transcript timing extends outside this recording; its speaker is not assigned.','La marca de tiempo de la transcripción queda fuera de esta grabación; no se le asigna un hablante.','文字起こしの時刻が録音の範囲外のため、話者は割り当てられていません。','转录时间超出本段录音范围，因此未分配说话人。'],
  'audio.result_details':['Speaker timing, alignment, limits and provenance','Tiempos, alineación, límites y procedencia de los hablantes','話者の時刻、対応付け、制限、出典','说话人时间、对齐、限制与来源']
});
Object.assign(workflowText,{
 'device.policy':['Device choice','Selección del dispositivo','デバイスの選択','设备选择'],
 'device.auto':['Automatic · compatible GPU preferred','Automático · priorizar GPU compatible','自動 · 対応GPUを優先','自动 · 优先兼容 GPU'],
 'device.cpu':['CPU · explicit choice','CPU · elección explícita','CPU · 明示的な選択','CPU · 明确选择'],
 'device.keepPinned':['Keep saved device assignment','Mantener la asignación guardada','保存済みの割り当てを維持','保留已保存的设备分配'],
 'device.checking':['Reading device selection…','Consultando la selección del dispositivo…','デバイス選択を確認中…','正在读取设备选择…'],
 'device.selected':['Selected device: {{device}}','Dispositivo seleccionado: {{device}}','選択デバイス：{{device}}','所选设备：{{device}}'],
 'device.missingSaved':['Saved device is unavailable: {{devices}}. No automatic CPU fallback.','El dispositivo guardado no está disponible: {{devices}}. No se cambia automáticamente a CPU.','保存済みデバイスが利用できません：{{devices}}。CPUへの自動切り替えは行いません。','已保存设备不可用：{{devices}}。不会自动回退到 CPU。'],
 'device.saved':['Saved devices: {{devices}}','Dispositivos guardados: {{devices}}','保存済みデバイス：{{devices}}','已保存设备：{{devices}}'],
 'device.recommended':['Automatic recommendation: {{device}}','Recomendación automática: {{device}}','自動選択の推奨：{{device}}','自动推荐：{{device}}'],
 'device.observed':['Observed loaded device: {{device}}','Dispositivo cargado observado: {{device}}','ロード済みの実測デバイス：{{device}}','已观察到的加载设备：{{device}}'],
 'device.notObserved':['Execution device has not been observed yet.','Aún no se ha observado el dispositivo de ejecución.','実行デバイスはまだ確認されていません。','尚未观察到实际执行设备。'],
 'device.selectionUnknown':['Device selection is unavailable or unverified. Refresh its metadata before loading.','La selección no está disponible o no está verificada. Actualice sus datos antes de cargar.','デバイス選択が利用不能または未確認です。ロード前に情報を更新してください。','设备选择不可用或尚未验证。加载前请刷新信息。'],
 'device.selectionNote':['Automatic prefers a compatible visible GPU. Selection and recommendations are not proof of execution.','El modo automático prioriza una GPU compatible y visible. La selección y las recomendaciones no prueban la ejecución.','自動では利用可能な対応GPUを優先します。選択や推奨は実行の証明ではありません。','自动模式优先选择可见的兼容 GPU。所选或推荐设备不等于实际执行证明。'],
 'device.maskNotice':['Server or guardian visibility is restricted. Hidden accelerators may still be installed; this does not mean the computer has no GPU.','El servidor o su supervisor limita la visibilidad. Puede haber aceleradores instalados pero ocultos; esto no significa que el equipo carezca de GPU.','サーバーまたは監視プロセスが可視性を制限しています。非表示でもGPUが搭載されている場合があります。','服务器或守护进程限制了设备可见性。隐藏的加速器可能仍已安装，这并不代表计算机没有 GPU。'],
 'device.unsavedNote':['Advanced device edits are unsaved. Save or discard that draft before changing the automatic/CPU policy.','Hay cambios avanzados sin guardar. Guárdelos o descártelos antes de cambiar la política automática/CPU.','詳細設定に未保存の変更があります。自動/CPU方針を変える前に保存または破棄してください。','高级设备设置尚未保存。更改自动/CPU 策略前，请先保存或放弃草稿。'],
 'device.refresh':['Refresh device information','Actualizar información de dispositivos','デバイス情報を更新','刷新设备信息'],
 'device.discard':['Discard assignment draft','Descartar el borrador de asignación','割り当ての下書きを破棄','放弃分配草稿'],
 'device.masked':['Hidden by server/guardian visibility settings','Oculto por la configuración del servidor/supervisor','サーバー/監視プロセスの設定により非表示','被服务器/守护进程的可见性设置隐藏'],
 'device.runtimeMissing':['Required device runtime is missing','Falta el runtime del dispositivo','必要なデバイスランタイムがありません','缺少所需的设备运行时'],
 'device.cpuOnly':['This adapter supports CPU only','Este adaptador solo admite CPU','このアダプターはCPU専用です','此适配器仅支持 CPU'],
 'device.weak':['No exact runtime mapping for this physical identity','No hay una correspondencia exacta del runtime para esta identidad física','この物理IDに対応する正確な実行時マッピングがありません','此物理标识缺少精确的运行时映射'],
 'device.unverified':['Runtime visibility or compatibility has not been verified','La visibilidad o compatibilidad del runtime no se ha verificado','実行環境の可視性または互換性は未確認です','运行时可见性或兼容性尚未验证'],
 'device.unavailable':['Saved device is unavailable','El dispositivo guardado no está disponible','保存済みデバイスが利用できません','已保存的设备不可用'],
 'device.incompatible':['Not available for this model/runtime','No disponible para este modelo/runtime','このモデル/ランタイムでは利用できません','此模型/运行时不支持'],
 'device.collective':['The required collective/runtime is unavailable on this platform','El sistema no dispone del runtime/colectivo necesario','この環境では必要な集団通信/ランタイムを利用できません','此平台缺少所需的集合通信库或运行时'],
 'device.unknownGroup':['Architecture mapping is unverified; a homogeneous group cannot be established','La arquitectura no está verificada; no se puede establecer un grupo homogéneo','アーキテクチャが未確認のため同種グループを確定できません','架构映射尚未验证，无法确认同构设备组'],
 'device.homogeneous':['Choose devices with the same runtime vendor and architecture','Elija dispositivos del mismo proveedor y arquitectura de runtime','同じランタイムベンダーとアーキテクチャのデバイスを選択してください','请选择运行时厂商和架构相同的设备'],
 'device.modeUnsupported':['This execution mode is not implemented for the selected adapter','Este modo no está implementado para el adaptador seleccionado','選択したアダプターではこの実行モードは未実装です','所选适配器尚未实现此执行模式'],
 'device.singleCount':['Choose one device, or select independent replicas or model splitting','Elija un dispositivo, réplicas independientes o división del modelo','デバイスを1つ選ぶか、独立レプリカまたはモデル分割を選択してください','请选择一个设备，或使用独立副本或模型拆分'],
 'device.twoDevices':['Select at least two compatible devices for this mode','Seleccione al menos dos dispositivos compatibles','このモードには対応デバイスを2つ以上選択してください','此模式至少需要两个兼容设备'],
 'device.replicaCount':['Replica count must be an integer from 1 to 16','El número de réplicas debe ser un entero entre 1 y 16','レプリカ数は1～16の整数です','副本数必须是 1 到 16 的整数'],
 'device.rankCount':['Rank count must be an integer from 2 to 16','El número de rangos debe ser un entero entre 2 y 16','ランク数は2～16の整数です','并行进程数必须是 2 到 16 的整数'],
 'device.rankMatch':['Choose one device for each tensor-parallel rank','Elija un dispositivo por rango de paralelismo tensorial','テンソル並列の各ランクにデバイスを1つ選択してください','请为每个张量并行进程选择一个设备'],
 'device.compatibilityNote':['Only compatible, visible mapped devices can be saved. Loading still validates resources and model support.','Solo se pueden guardar dispositivos compatibles, visibles y mapeados. La carga verifica recursos y compatibilidad del modelo.','対応していて可視かつマッピング済みのデバイスのみ保存できます。ロード時に資源とモデル互換性を検証します。','只能保存兼容、可见且已映射的设备。加载时仍会验证资源和模型支持。'],
 'device.inventoryDetails':['Device visibility and compatibility details','Detalles de visibilidad y compatibilidad','デバイスの可視性と互換性の詳細','设备可见性与兼容性详情'],
 'device.invalidInventory':['Invalid device inventory response','Respuesta de inventario de dispositivos no válida','デバイス一覧の応答が不正です','设备清单响应无效'],
 'device.nativeTensor':['Native tensor split · experimental','División tensorial nativa · experimental','ネイティブテンソル分割 · 実験的','原生张量拆分 · 实验性'],
 'device.nativeShard':['Native layers/rows across devices','Capas/filas nativas entre dispositivos','デバイス間のネイティブ層/行分割','跨设备原生层/行拆分'],
 'device.modeReason':['{{mode}}: {{reason}}','{{mode}}: {{reason}}','{{mode}}：{{reason}}','{{mode}}：{{reason}}'],
 'device.runtimeUnavailable':['The selected runtime profile is unavailable. Choose a compatible profile.','El perfil de runtime seleccionado no está disponible. Elija uno compatible.','選択したランタイムプロファイルは利用できません。対応するプロファイルを選択してください。','所选运行时配置不可用。请选择兼容配置。'],
 'device.profileMismatch':['A CPU runtime cannot satisfy an explicit accelerator selection. Change the profile or device.','Un runtime CPU no puede usar un acelerador explícito. Cambie el perfil o dispositivo.','CPUランタイムでは明示したアクセラレーターを使えません。プロファイルまたはデバイスを変更してください。','CPU 运行时无法满足明确的加速器选择。请更改配置或设备。'],
 'device.prepareRuntime':['Prepare selected runtime, then download and run','Preparar el runtime seleccionado, descargar y ejecutar','選択したランタイムを準備し、ダウンロードして実行','准备所选运行时，然后下载并运行']
});
Object.assign(workflowText,{
 'device.replan':['Device or runtime selection changed. Review the model plan again before downloading.','Cambió la selección de dispositivo o runtime. Revise el plan antes de descargar.','デバイスまたはランタイムが変更されました。ダウンロード前にモデル計画を再確認してください。','设备或运行时选择已更改。下载前请重新查看模型准备计划。'],
 'device.pendingRuntime':['Another runtime profile is still being prepared. Finish or cancel it before changing profiles.','Se está preparando otro perfil de runtime. Finalícelo o cancélelo antes de cambiar de perfil.','別のランタイムプロファイルを準備中です。完了またはキャンセルしてから変更してください。','另一个运行时配置仍在准备中。请先完成或取消，再切换配置。'],
 'device.activeProfileMismatch':['This model is already registered with another runtime profile. Review its registration; it was not reloaded or migrated.','Este modelo ya está registrado con otro perfil de runtime. Revise su registro; no se volvió a cargar ni se migró.','このモデルは別のランタイムプロファイルで登録済みです。登録を確認してください。再ロードや移行はしていません。','此模型已使用其他运行时配置注册。请检查注册；未重新加载或迁移。'],
 'device.activeProfileUnknown':['This existing registration did not record its runtime profile. Its match to the selected profile is unknown; it was not reloaded.','Este registro no guardó su perfil de runtime. No se sabe si coincide con el seleccionado; no se volvió a cargar.','既存の登録にランタイムプロファイルの記録がありません。選択との一致は不明です。再ロードはしていません。','现有注册未记录运行时配置，无法确认是否匹配所选配置；未重新加载。'],
 'device.runtimeReady':['Selected managed runtime is verified and ready.','El runtime seleccionado está verificado y listo.','選択した管理ランタイムの検証が完了しました。','所选托管运行时已验证并就绪。'],
 'device.preparingRuntime':['Preparing selected managed runtime','Preparando el runtime seleccionado','選択した管理ランタイムを準備中','正在准备所选托管运行时'],
 'device.planningRuntime':['Planning the selected server-owned runtime…','Planificando el runtime seleccionado del servidor…','選択したサーバー管理ランタイムを計画中…','正在规划所选服务器管理的运行时…']
});
// Studio 'Cloud and routing' panel, location select, external badges and provenance line (broker.*)
Object.assign(workflowText,{
 'broker.activity':["Usage and recent decisions", "Uso y decisiones recientes", "使用量と最近の判断", "用量与最近的决策"],
 'broker.alias_name':["Published name", "Nombre publicado", "公開名", "发布名称"],
 'broker.alias_row':["{{name}} → {{target}}", "{{name}} → {{target}}", "{{name}} → {{target}}", "{{name}} → {{target}}"],
 'broker.alias_target':["Target", "Destino", "ターゲット", "目标"],
 'broker.allow_http_lan':["Allow plain http to LAN upstreams", "Permitir http sin cifrar hacia upstreams de la LAN", "LAN 上のアップストリームへの平文 http を許可", "允许以明文 http 连接局域网上游"],
 'broker.anonymous_cloud':["Allow cloud for keyless loopback callers", "Permitir la nube a llamadas loopback sin clave", "キーなしのループバック呼び出しにクラウドを許可", "允许无密钥的本机回环调用使用云端"],
 'broker.badge_external':["External · {{upstream}}", "Externo · {{upstream}}", "外部 · {{upstream}}", "外部 · {{upstream}}"],
 'broker.badge_virtual':["Virtual", "Virtual", "仮想", "虚拟"],
 'broker.bind_kept':["Saved. Upstream {{upstream}} still uses {{current}}.", "Guardado. El upstream {{upstream}} sigue usando {{current}}.", "保存しました。アップストリーム {{upstream}} は引き続き {{current}} を使用します。", "已保存。上游 {{upstream}} 仍使用 {{current}}。"],
 'broker.bind_needed':["Saved. Upstream {{upstream}} does not use {{id}} yet; a key with broker:admin must bind it.", "Guardado. El upstream {{upstream}} aún no usa {{id}}; una clave con broker:admin debe vincularla.", "保存しました。アップストリーム {{upstream}} はまだ {{id}} を使用していません。broker:admin を持つキーで関連付けてください。", "已保存。上游 {{upstream}} 尚未使用 {{id}}；需要具有 broker:admin 的密钥来绑定。"],
 'broker.blank_unchanged':["The value was blank; nothing changed.", "El valor estaba vacío; no se cambió nada.", "値が空だったため、何も変更していません。", "值为空，未做任何更改。"],
 'broker.bound':["Saved and bound to {{upstream}}. The credential value is not shown again.", "Guardado y vinculado a {{upstream}}. El valor de la credencial no se vuelve a mostrar.", "保存し、{{upstream}} に関連付けました。資格情報の値は再表示されません。", "已保存并绑定到 {{upstream}}。凭据值不会再次显示。"],
 'broker.capability':["Capability", "Capacidad", "機能", "能力"],
 'broker.cloud_explicit':["Explicit models only", "Solo modelos explícitos", "明示したモデルのみ", "仅限显式指定的模型"],
 'broker.cloud_fallback':["Explicit and fallback", "Explícitos y respaldo", "明示と代替", "显式与回退"],
 'broker.cloud_off':["Off", "Desactivada", "オフ", "关闭"],
 'broker.cloud_policy':["Cloud policy", "Política de nube", "クラウドポリシー", "云端策略"],
 'broker.col_cost':["Cost (USD)", "Costo (USD)", "コスト (USD)", "费用 (USD)"],
 'broker.col_fallbacks':["Fallbacks", "Respaldos", "代替回数", "回退次数"],
 'broker.col_model':["Model", "Modelo", "モデル", "模型"],
 'broker.col_requests':["Requests", "Solicitudes", "リクエスト", "请求数"],
 'broker.col_served':["Served by", "Atendido por", "処理元", "提供方"],
 'broker.col_time':["Time", "Hora", "時刻", "时间"],
 'broker.col_upstream':["Upstream", "Upstream", "アップストリーム", "上游"],
 'broker.confirm_alias':["Remove {{name}}?", "¿Quitar {{name}}?", "{{name}} を削除しますか?", "移除 {{name}}？"],
 'broker.confirm_bind':["Upstream {{upstream}} uses credential {{current}}. Switch it to {{id}}?", "El upstream {{upstream}} usa la credencial {{current}}. ¿Cambiarla a {{id}}?", "アップストリーム {{upstream}} は資格情報 {{current}} を使用しています。{{id}} に切り替えますか?", "上游 {{upstream}} 使用凭据 {{current}}。要切换为 {{id}} 吗？"],
 'broker.confirm_remove':["Remove credential {{id}}? Upstreams using it stop working.", "¿Quitar la credencial {{id}}? Los upstreams que la usan dejarán de funcionar.", "資格情報 {{id}} を削除しますか? これを使うアップストリームは動作しなくなります。", "移除凭据 {{id}}？使用它的上游将停止工作。"],
 'broker.confirm_revoke':["Revoke {{id}}? This cannot be undone.", "¿Revocar {{id}}? No se puede deshacer.", "{{id}} を失効させますか? 元に戻せません。", "吊销 {{id}}？此操作无法撤销。"],
 'broker.confirm_rotate':["Rotate {{id}}? The current secret stops working at once.", "¿Rotar {{id}}? El secreto actual deja de funcionar de inmediato.", "{{id}} をローテーションしますか? 現在のシークレットはすぐに無効になります。", "轮换 {{id}}？当前密钥将立即失效。"],
 'broker.copy':["Copy", "Copiar", "コピー", "复制"],
 'broker.create_key':["Create key", "Crear clave", "キーを作成", "创建密钥"],
 'broker.credential_expected':["Leave the id blank to use {{id}}.", "Deja el id en blanco para usar {{id}}.", "ID を空欄にすると {{id}} を使用します。", "将 ID 留空即使用 {{id}}。"],
 'broker.credential_id':["Credential id", "Id de la credencial", "資格情報 ID", "凭据 ID"],
 'broker.credential_id_hint':["credential id, e.g. openai-key", "id de la credencial, p. ej. openai-key", "資格情報 ID (例: openai-key)", "凭据 ID，例如 openai-key"],
 'broker.credential_id_needed':["Enter the credential id to remove.", "Escriba el id de la credencial que desea quitar.", "削除する資格情報 ID を入力してください。", "请输入要移除的凭据 ID。"],
 'broker.credentials':["Add or replace credential", "Añadir o reemplazar credencial", "資格情報の追加または置換", "添加或替换凭据"],
 'broker.default_location':["Default placement", "Ubicación predeterminada", "既定の実行場所", "默认运行位置"],
 'broker.disabled':["Disabled", "Desactivado", "無効", "已停用"],
 'broker.dismiss':["Done, hide it", "Listo, ocultar", "完了して非表示", "完成并隐藏"],
 'broker.enabled':["Enabled", "Activado", "有効", "已启用"],
 'broker.env':["Environment variable on the server host", "Variable de entorno en el servidor", "サーバーホストの環境変数", "服务器主机上的环境变量"],
 'broker.explain':["Explain routing", "Explicar el enrutamiento", "ルーティングを説明", "说明路由"],
 'broker.explain_model':["Model to explain", "Modelo que explicar", "説明するモデル", "要说明的模型"],
 'broker.fallback_on':["Fall back on", "Recurrir al respaldo ante", "代替する条件", "回退条件"],
 'broker.file':["Owner-only file path on the server host", "Ruta de un archivo solo del propietario en el servidor", "サーバーホスト上の所有者専用ファイルのパス", "服务器主机上仅所有者可读的文件路径"],
 'broker.insecure':["Credential writes need https or a loopback address; this page is neither, so the form is disabled.", "Guardar credenciales requiere https o una dirección loopback; esta página no usa ninguna, así que el formulario está desactivado.", "資格情報の書き込みには https かループバックアドレスが必要です。このページはどちらでもないため、フォームは無効です。", "写入凭据需要 https 或本机回环地址；此页面两者都不是，因此表单已停用。"],
 'broker.key_budget':["Budget per day (USD)", "Presupuesto diario (USD)", "1 日の予算 (USD)", "每日预算 (USD)"],
 'broker.key_days':["Expires after days", "Caduca tras días", "有効日数", "有效天数"],
 'broker.key_meta':["{{id}} · {{status}} · {{scopes}}", "{{id}} · {{status}} · {{scopes}}", "{{id}} · {{status}} · {{scopes}}", "{{id}} · {{status}} · {{scopes}}"],
 'broker.key_models':["Models", "Modelos", "モデル", "模型"],
 'broker.key_name':["Key name", "Nombre de la clave", "キー名", "密钥名称"],
 'broker.key_rpm':["Requests per minute", "Solicitudes por minuto", "1 分あたりのリクエスト数", "每分钟请求数"],
 'broker.keyless_note':["Keyless mode cannot write credentials from Studio. On this computer run: titanos broker credential set <id> --upstream <upstream> (the value is read from a hidden prompt), or create a key with titanos keys create to enable this form.", "El modo sin claves no puede guardar credenciales desde Studio. En este equipo ejecute: titanos broker credential set <id> --upstream <upstream> (el valor se lee de un aviso oculto), o cree una clave con titanos keys create para activar este formulario.", "キーなしモードでは Studio から資格情報を書き込めません。このコンピューターで titanos broker credential set <id> --upstream <upstream> を実行してください (値は非表示のプロンプトで入力します)。または titanos keys create でキーを作成すると、このフォームが有効になります。", "无密钥模式下无法从 Studio 写入凭据。请在本机运行：titanos broker credential set <id> --upstream <upstream>（值从隐藏提示中读取），或使用 titanos keys create 创建密钥以启用此表单。"],
 'broker.keyless_policy':["Keyless callers cannot change these two switches; edit broker.json on the host.", "Las llamadas sin clave no pueden cambiar estos dos interruptores; edite broker.json en el equipo.", "キーなしの呼び出しではこの 2 つのスイッチを変更できません。ホストで broker.json を編集してください。", "无密钥调用无法更改这两个开关；请在主机上编辑 broker.json。"],
 'broker.keys':["Keys", "Claves", "キー", "密钥"],
 'broker.location':["Where to run", "Dónde ejecutar", "実行場所", "运行位置"],
 'broker.location_auto':["Automatic", "Automático", "自動", "自动"],
 'broker.location_cloud_only':["Cloud only", "Solo nube", "クラウドのみ", "仅云端"],
 'broker.location_default':["Server default", "Predeterminado del servidor", "サーバーの既定", "服务器默认"],
 'broker.location_local_only':["This computer only", "Solo este equipo", "このコンピューターのみ", "仅本机"],
 'broker.location_prefer_local':["Prefer this computer", "Preferir este equipo", "このコンピューターを優先", "优先本机"],
 'broker.max_attempts':["Maximum attempts", "Intentos máximos", "最大試行回数", "最大尝试次数"],
 'broker.metadata_only':["Metadata only; prompts and outputs are never stored.", "Solo metadatos; las solicitudes y respuestas nunca se guardan.", "メタデータのみです。プロンプトと出力は保存されません。", "仅元数据；从不存储提示和输出。"],
 'broker.models_hint':["Model globs, comma separated (default all)", "Patrones de modelo separados por comas (todos por defecto)", "モデルのパターン (カンマ区切り、既定はすべて)", "模型通配符，逗号分隔（默认全部）"],
 'broker.no_upstreams':["No upstreams are configured.", "No hay upstreams configurados.", "アップストリームは構成されていません。", "尚未配置上游。"],
 'broker.policy':["Policy", "Política", "ポリシー", "策略"],
 'broker.policy_summary':["Cloud policy {{effective}} (file {{file}}, ceiling {{ceiling}})", "Política de nube {{effective}} (archivo {{file}}, límite {{ceiling}})", "クラウドポリシー {{effective}} (ファイル {{file}}、上限 {{ceiling}})", "云端策略 {{effective}}（文件 {{file}}，上限 {{ceiling}}）"],
 'broker.provenance':["Served by {{served}} · {{model}} · fallbacks {{fallbacks}}", "Atendido por {{served}} · {{model}} · respaldos {{fallbacks}}", "処理元 {{served}} · {{model}} · 代替 {{fallbacks}} 回", "由 {{served}} 提供 · {{model}} · 回退 {{fallbacks}} 次"],
 'broker.remove':["Remove", "Quitar", "削除", "移除"],
 'broker.remove_credential':["Remove credential", "Quitar credencial", "資格情報を削除", "移除凭据"],
 'broker.removed':["Removed.", "Quitado.", "削除しました。", "已移除。"],
 'broker.revoke':["Revoke", "Revocar", "失効", "吊销"],
 'broker.role':["Role: {{role}}", "Rol: {{role}}", "ロール: {{role}}", "角色：{{role}}"],
 'broker.rotate':["Rotate", "Rotar", "ローテーション", "轮换"],
 'broker.routing':["Aliases and virtual models", "Alias y modelos virtuales", "エイリアスと仮想モデル", "别名与虚拟模型"],
 'broker.save':["Save credential", "Guardar credencial", "資格情報を保存", "保存凭据"],
 'broker.save_alias':["Save alias", "Guardar alias", "エイリアスを保存", "保存别名"],
 'broker.save_policy':["Save policy", "Guardar política", "ポリシーを保存", "保存策略"],
 'broker.save_virtual':["Save virtual model", "Guardar modelo virtual", "仮想モデルを保存", "保存虚拟模型"],
 'broker.saved':["Saved. The credential value is not shown again.", "Guardado. El valor de la credencial no se vuelve a mostrar.", "保存しました。資格情報の値は再表示されません。", "已保存。凭据值不会再次显示。"],
 'broker.scopes':["Scopes", "Ámbitos", "スコープ", "权限范围"],
 'broker.secret':["Secret value (write-only)", "Valor secreto (solo escritura)", "シークレット値 (書き込み専用)", "密钥值（只写）"],
 'broker.secret_once':["Key {{id}} created. Copy this secret now; it is shown only once.", "Clave {{id}} creada. Copie este secreto ahora; solo se muestra una vez.", "キー {{id}} を作成しました。このシークレットを今すぐコピーしてください。表示は一度だけです。", "已创建密钥 {{id}}。请立即复制此密钥值；它只显示一次。"],
 'broker.secret_self':["Key {{id}} rotated. This tab now uses the new secret; copy it now, it is shown only once.", "Clave {{id}} rotada. Esta pestaña ya usa el nuevo secreto; cópielo ahora, solo se muestra una vez.", "キー {{id}} をローテーションしました。このタブは新しいシークレットを使用しています。今すぐコピーしてください。表示は一度だけです。", "已轮换密钥 {{id}}。此标签页现在使用新的密钥值；请立即复制，它只显示一次。"],
 'broker.source':["Source", "Origen", "ソース", "来源"],
 'broker.source_env':["Environment variable", "Variable de entorno", "環境変数", "环境变量"],
 'broker.source_file':["File path", "Ruta de archivo", "ファイルパス", "文件路径"],
 'broker.source_secret':["Secret value", "Valor secreto", "シークレット値", "密钥值"],
 'broker.strategy':["Strategy", "Estrategia", "戦略", "策略方式"],
 'broker.strategy_cheapest':["Cheapest", "Más barata", "最安", "最便宜"],
 'broker.strategy_lowest_latency':["Lowest latency", "Menor latencia", "最小遅延", "最低延迟"],
 'broker.strategy_ordered':["In order", "En orden", "順番", "按顺序"],
 'broker.strategy_weighted':["Weighted", "Ponderada", "重み付け", "加权"],
 'broker.target_hint':["model@upstream", "modelo@upstream", "モデル@アップストリーム", "模型@上游"],
 'broker.targets':["Ordered targets", "Destinos en orden", "順序付きターゲット", "有序目标"],
 'broker.targets_hint':["One target per line in order, e.g. qwen3-8b then gpt-4o-mini@openai", "Un destino por línea en orden, p. ej. qwen3-8b y luego gpt-4o-mini@openai", "1 行に 1 つのターゲットを順番に (例: qwen3-8b、次に gpt-4o-mini@openai)", "每行一个目标，按顺序，例如先 qwen3-8b 再 gpt-4o-mini@openai"],
 'broker.test':["Test", "Probar", "テスト", "测试"],
 'broker.test_failed':["Test failed: {{code}}", "La prueba falló: {{code}}", "テストに失敗しました: {{code}}", "测试失败：{{code}}"],
 'broker.test_ok':["Test passed in {{ms}} ms", "Prueba superada en {{ms}} ms", "テストに成功しました ({{ms}} ms)", "测试通过，用时 {{ms}} 毫秒"],
 'broker.title':["Cloud and routing", "Nube y enrutamiento", "クラウドとルーティング", "云端与路由"],
 'broker.upstream':["Upstream", "Upstream", "アップストリーム", "上游"],
 'broker.upstream_credential':["Credential {{fingerprint}} · validated {{validated}}", "Credencial {{fingerprint}} · validada {{validated}}", "資格情報 {{fingerprint}} · 検証 {{validated}}", "凭据 {{fingerprint}} · 验证于 {{validated}}"],
 'broker.upstream_meta':["{{profile}} · {{label}}", "{{profile}} · {{label}}", "{{profile}} · {{label}}", "{{profile}} · {{label}}"],
 'broker.upstream_state':["{{enabled}} · breaker {{breaker}} · catalog {{catalog}}", "{{enabled}} · disyuntor {{breaker}} · catálogo {{catalog}}", "{{enabled}} · ブレーカー {{breaker}} · カタログ {{catalog}}", "{{enabled}} · 熔断器 {{breaker}} · 目录 {{catalog}}"],
 'broker.upstreams':["Upstreams", "Upstreams", "アップストリーム", "上游"],
 'broker.vault_open':["Vault {{backend}} · open", "Almacén {{backend}} · abierto", "ボールト {{backend}} · 利用可能", "保管库 {{backend}} · 开放"],
 'broker.vault_sealed':["Vault {{backend}} · sealed", "Almacén {{backend}} · sellado", "ボールト {{backend}} · 封印中", "保管库 {{backend}} · 已封存"],
 'broker.virtual_name':["Virtual model name", "Nombre del modelo virtual", "仮想モデル名", "虚拟模型名称"],
 'broker.virtual_row':["{{name}} · {{strategy}} · {{targets}}", "{{name}} · {{strategy}} · {{targets}}", "{{name}} · {{strategy}} · {{targets}}", "{{name}} · {{strategy}} · {{targets}}"]
});
// Jobs section state for a key without the 'jobs' scope (GET /api/jobs answers 403); not a connection failure
Object.assign(workflowText,{
 'jobs.unavailable':["Jobs are not available for this key. Connect with a key that has the jobs scope to list and run jobs.", "Los trabajos no están disponibles para esta clave. Conéctate con una clave que tenga el ámbito jobs para ver y ejecutar trabajos.", "このキーではジョブを利用できません。ジョブの一覧表示と実行には、jobs スコープを持つキーで接続してください。", "此密钥无法使用任务。请使用具有 jobs 范围的密钥连接，以查看和运行任务。"]
});
// Studio cabinet: bays, readouts, receipts, the boot readout, jobs scoreboard and the provider connect flow (R29-R31).
Object.assign(workflowText,{
 'studio.plate':["Studio", "Studio", "スタジオ", "工作室"],
 "error.unreachable":["Titanos is not reachable at {{host}}. Start it or press Refresh.", "No se puede contactar con Titanos en {{host}}. Inícielo o pulse Actualizar.", "{{host}} の Titanos に接続できません。起動するか、更新を押してください。", "无法连接 {{host}} 上的 Titanos。请启动它或点击刷新。"],
 "run.ready":["Ready. Press Run.", "Listo. Pulse Ejecutar.", "準備完了。実行を押してください。", "已就绪。请点击运行。"],
 "run.unloaded":["Not loaded. Load it first, or Run loads it if this server allows.", "Sin cargar. Cárguelo primero, o Ejecutar lo carga si este servidor lo permite.", "未読み込みです。先に読み込むか、サーバーが許可していれば実行時に読み込まれます。", "未加载。请先加载；如果服务器允许，点击运行时会自动加载。"],
 "connection.connected_one":["Connected, {{count}} model", "Conectado, {{count}} modelo", "接続済み、{{count}} モデル", "已连接，{{count}} 个模型"],
 "models.technical_for":["Details for {{model}}", "Detalles de {{model}}", "{{model}} の詳細", "{{model}} 的详情"],
 "results.count_one":["{{count}} result", "{{count}} resultado", "結果 {{count}} 件", "{{count}} 个结果"],
 "jobs.cancelled_count":["{{count}} cancelled", "{{count}} cancelados", "キャンセル {{count}}", "已取消 {{count}}"],
 "rail.more":["{{count}} more below", "{{count}} más abajo", "下にあと {{count}} 件", "下方还有 {{count}} 个"],
 "ui.vector_preview_one":["{{count}} vector · {{dimensions}} dimensions\nPreview: {{preview}}", "{{count}} vector · {{dimensions}} dimensiones\nVista previa: {{preview}}", "{{count}} ベクトル · {{dimensions}} 次元\nプレビュー: {{preview}}", "{{count}} 个向量 · {{dimensions}} 维\n预览：{{preview}}"],
 "ui.detection_summary_one":["{{count}} detected object{{size}}. Native scores are not verified truth.", "{{count}} objeto detectado{{size}}. Las puntuaciones nativas no son hechos verificados.", "検出物体 {{count}} 個{{size}}。ネイティブスコアは検証済みの事実ではありません。", "检测到 {{count}} 个对象{{size}}。原生分数不是已验证的事实。"],
 "ui.vad_result_one":["{{count}} speech interval · {{kind}} · uncalibrated.", "{{count}} intervalo de voz · {{kind}} · sin calibrar.", "音声区間 {{count}} 件 · {{kind}} · 未校正。", "{{count}} 个语音区间 · {{kind}} · 未经校准。"],
 "ui.replica_summary_one":["{{mode}} · {{count}} replica", "{{mode}} · {{count}} réplica", "{{mode}} · レプリカ {{count}} 個", "{{mode}} · {{count}} 个副本"],
 "ui.replica_saved_one":["{{mode}} · {{count}} replica saved", "{{mode}} · {{count}} réplica guardada", "{{mode}} · レプリカ {{count}} 個を保存", "{{mode}} · 已保存 {{count}} 个副本"],
 "ui.found_candidates_one":["Found {{count}} candidate. Review compatibility before downloading.", "Se encontró {{count}} candidato. Revisa su compatibilidad antes de descargar.", "候補が {{count}} 件見つかりました。ダウンロード前に互換性を確認してください。", "找到 {{count}} 个候选模型。下载前请检查兼容性。"],
 "ui.selected_files_one":["{{count}} selected file", "{{count}} archivo seleccionado", "選択したファイル {{count}} 件", "已选择 {{count}} 个文件"],
 "ui.devices_loaded_one":["Loaded {{count}} known device record. Inventory presence is not inference qualification.", "Se cargó {{count}} dispositivo conocido. Estar en el inventario no acredita la inferencia.", "既知のデバイス {{count}} 件を読み込みました。一覧への存在は推論の検証を意味しません。", "已加载 {{count}} 条已知设备记录。清单中存在设备并不代表推理已获验证。"],
 "broker.catalog_value_one":["{{source}} · {{count}} model", "{{source}} · {{count}} modelo", "{{source}} · モデル {{count}} 件", "{{source}} · {{count}} 个模型"],
 "broker.connected_one":["Connected {{provider}}: {{count}} model found. It is listed under Models, marked External.", "{{provider}} conectado: {{count}} modelo encontrado. Aparece en Modelos, marcado como Externo.", "{{provider}} を接続しました: モデル {{count}} 件。「モデル」に外部として表示されます。", "已连接 {{provider}}：找到 {{count}} 个模型，已在“模型”中标为外部。"],
 "broker.connect_option_idle":["{{name}} · configured, not enabled", "{{name}} · configurado, sin habilitar", "{{name}} · 設定済み、無効", "{{name}} · 已配置，未启用"],
 "broker.connect_idle":["Upstream {{id}} is configured but not enabled. Connecting checks the key, binds it and enables the upstream.", "El upstream {{id}} está configurado pero no habilitado. Conectar comprueba la clave, la vincula y habilita el upstream.", "アップストリーム {{id}} は設定済みですが無効です。接続するとキーを確認して割り当て、有効にします。", "上游 {{id}} 已配置但未启用。连接会检查密钥、绑定并启用该上游。"],
 "broker.connect_option_unready":["{{name}} · configured, not ready", "{{name}} · configurado, no listo", "{{name}} · 設定済み、未準備", "{{name}} · 已配置，未就绪"],
 "broker.connect_unready":["Upstream {{id}} is enabled but not ready: its key is not validated, or the server cannot read it from its own environment. Connecting checks the key from the server, binds it and refreshes its models.", "El upstream {{id}} está habilitado pero no está listo: su clave no está validada o el servidor no puede leerla de su propio entorno. Conectar comprueba la clave desde el servidor, la vincula y actualiza sus modelos.", "アップストリーム {{id}} は有効ですが準備ができていません。キーが検証されていないか、サーバーが自身の環境からキーを読み取れません。接続するとサーバーからキーを確認して割り当て、モデルを更新します。", "上游 {{id}} 已启用但未就绪：其密钥未经验证，或服务器无法从自身环境读取该密钥。连接会从服务器检查密钥、绑定密钥并刷新其模型。"],
 "broker.step_ceiling":["Cloud calls are refused while the cloud policy or its ceiling is off. Raise them to use {{upstream}}.", "Las llamadas a la nube se rechazan mientras la política de nube o su límite estén desactivados. Súbalos para usar {{upstream}}.", "クラウドポリシーまたはその上限がオフの間、クラウド呼び出しは拒否されます。{{upstream}} を使うには引き上げてください。", "云端策略或其上限关闭时，云端调用会被拒绝。请调高后再使用 {{upstream}}。"],
 "broker.step_keyless":["Keyless programs on this computer, Studio included, reach {{upstream}} only when keyless cloud use is allowed (policy, Advanced).", "Los programas sin clave de este equipo, Studio incluido, llegan a {{upstream}} solo si se permite el uso de la nube sin clave (política, Avanzado).", "Studio を含むこのコンピューター上のキーなしのプログラムは、キーなしのクラウド利用が許可されている場合にのみ {{upstream}} を使えます（ポリシー、詳細モード）。", "本机上的无密钥程序（包括 Studio）只有在允许无密钥云端使用时才能访问 {{upstream}}（策略，高级模式）。"],
 "broker.step_call":["Its models are listed under Models with their full ids, for example {{example}}.", "Sus modelos aparecen en Modelos con su id completo, por ejemplo {{example}}.", "モデルは「モデル」に完全な ID で表示されます（例: {{example}}）。", "其模型以完整 ID 列在“模型”中，例如 {{example}}。"],
 "broker.usage_empty":["No cloud requests recorded yet.", "Aún no hay solicitudes a la nube registradas.", "クラウドへのリクエストはまだ記録されていません。", "尚未记录云端请求。"],
 "broker.decisions_empty":["No routing decisions recorded yet.", "Aún no hay decisiones de enrutamiento registradas.", "ルーティングの判断はまだ記録されていません。", "尚未记录路由决策。"],
 "retrieval.models_cell":["Models", "Modelos", "モデル", "模型"],
 "retrieval.passages_cell":["Passages and question", "Pasajes y pregunta", "文章と質問", "段落与问题"],
 'index.label':["Sections", "Secciones", "セクション", "分区"],
 'results.title':["Results", "Resultados", "結果", "结果"],
 'results.count':["{{count}} results", "{{count}} resultados", "結果 {{count}} 件", "{{count}} 个结果"],
 'results.empty':["Nothing recorded yet. Each result lands here with a receipt: model, route, time and size.", "Aún no hay nada registrado. Cada resultado llega aquí con un recibo: modelo, ruta, tiempo y tamaño.", "まだ記録はありません。結果はモデル、経路、時間、サイズを示す控えとともにここに表示されます。", "尚无记录。每个结果都会带着回执出现在这里：模型、路由、时间和大小。"],
 'results.empty_first':["No cells filled yet. Add a model in 02, or connect a provider in 07.", "Aún no hay celdas llenas. Añada un modelo en 02 o conecte un proveedor en 07.", "まだセルは空です。02 でモデルを追加するか、07 でプロバイダーを接続してください。", "还没有填充的单元。请在 02 添加模型，或在 07 连接提供商。"],
 'rail.cloud_routes':["Cloud routes", "Rutas en la nube", "クラウド経路", "云端路由"],
 'rail.local':["{{count}} local", "{{count}} locales", "ローカル {{count}}", "本地 {{count}}"],
 'rail.loaded':["{{count}} loaded", "{{count}} cargados", "読み込み済み {{count}}", "已加载 {{count}}"],
 'rail.cloud':["{{count}} cloud", "{{count}} en la nube", "クラウド {{count}}", "云端 {{count}}"],
 'rail.position':["{{index}} of {{count}}", "{{index}} de {{count}}", "{{count}} 件中 {{index}}", "第 {{index}} / {{count}}"],
 'ui.model_title_cloud':["{{model}} · {{task}} · {{provider}}", "{{model}} · {{task}} · {{provider}}", "{{model}} · {{task}} · {{provider}}", "{{model}} · {{task}} · {{provider}}"],
 'ui.route_note':["This route leaves your machine. Titanos sends only what you run here.", "Esta ruta sale de su equipo. Titanos solo envía lo que ejecute aquí.", "この経路はお使いのコンピューターの外へ出ます。Titanos が送るのはここで実行した内容だけです。", "此路由会离开您的计算机。Titanos 只发送您在这里运行的内容。"],
 'readout.route':["Route", "Ruta", "経路", "路由"],
 'readout.route_local':["This computer", "Este equipo", "このコンピューター", "本机"],
 'readout.route_cloud':["Cloud · {{provider}}", "Nube · {{provider}}", "クラウド · {{provider}}", "云端 · {{provider}}"],
 'readout.cloud':["Cloud", "Nube", "クラウド", "云端"],
 'readout.class':["Class", "Clase", "種別", "类别"],
 'readout.cloud_model':["Cloud model", "Modelo en la nube", "クラウドモデル", "云端模型"],
 'readout.task':["Task", "Tarea", "タスク", "任务"],
 'readout.state':["State", "Estado", "状態", "状态"],
 'readout.device':["Device", "Dispositivo", "デバイス", "设备"],
 'readout.backend':["Backend", "Backend", "バックエンド", "后端"],
 'readout.mode':["Mode", "Modo", "モード", "模式"],
 'readout.source':["Source", "Origen", "ソース", "来源"],
 'receipt.recorded':["Recorded result", "Resultado registrado", "記録された結果", "已记录的结果"],
 'receipt.failed':["Failed", "Fallido", "失敗", "失败"],
 'receipt.external':["External", "Externo", "外部", "外部"],
 'receipt.seconds':["{{value}} s", "{{value}} s", "{{value}} 秒", "{{value}} 秒"],
 'receipt.tokens':["{{count}} tokens", "{{count}} tokens", "{{count}} トークン", "{{count}} 个令牌"],
 'receipt.tokens_unit':["tokens", "tokens", "トークン", "令牌"],
 'receipt.chunks':["{{count}} chunks", "{{count}} fragmentos", "{{count}} チャンク", "{{count}} 个分块"],
 'receipt.chunks_unit':["chunks", "fragmentos", "チャンク", "分块"],
 'receipt.characters':["{{count}} characters", "{{count}} caracteres", "{{count}} 文字", "{{count}} 个字符"],
 'receipt.recorded_at':["Recorded {{time}}", "Registrado {{time}}", "記録 {{time}}", "记录于 {{time}}"],
 'receipt.cost':["${{amount}}", "{{amount}} US$", "${{amount}}", "${{amount}}"],
 'jobs.col_state':["State", "Estado", "状態", "状态"],
 'jobs.col_started':["Started", "Inicio", "開始", "开始"],
 'jobs.col_job':["Job", "Trabajo", "ジョブ", "作业"],
 'jobs.col_task':["Task", "Tarea", "タスク", "任务"],
 'jobs.col_model':["Model", "Modelo", "モデル", "模型"],
 'jobs.col_time':["Time", "Tiempo", "時間", "用时"],
 'jobs.col_actions':["Actions", "Acciones", "操作", "操作"],
 'jobs.running_count':["{{count}} running", "{{count}} en curso", "実行中 {{count}}", "运行中 {{count}}"],
 'jobs.queued_count':["{{count}} queued", "{{count}} en cola", "待機中 {{count}}", "排队中 {{count}}"],
 'jobs.done_count':["{{count}} done", "{{count}} terminados", "完了 {{count}}", "已完成 {{count}}"],
 'jobs.failed_count':["{{count}} failed", "{{count}} fallidos", "失敗 {{count}}", "失败 {{count}}"],
 'boot.ready':["> Titanos_ready", "> Titanos_listo", "> Titanos_準備完了", "> Titanos_就绪"],
 'boot.offline':["> Connection needed", "> Conexión necesaria", "> 接続が必要です", "> 需要连接"],
 'boot.waiting':["> Booting", "> Iniciando", "> 起動中", "> 启动中"],
 'boot.engine':["Engine", "Motor", "エンジン", "引擎"],
 'boot.engine_ok':["OK · v{{version}}", "OK · v{{version}}", "OK · v{{version}}", "正常 · v{{version}}"],
 'boot.ok':["OK", "OK", "OK", "正常"],
 'boot.models':["Models", "Modelos", "モデル", "模型"],
 'boot.cloud':["Cloud", "Nube", "クラウド", "云端"],
 'boot.policy':["policy {{policy}}", "política {{policy}}", "ポリシー {{policy}}", "策略 {{policy}}"],
 'boot.cloud_off':["Off · nothing leaves this computer", "Desactivada · nada sale de este equipo", "オフ · このコンピューターの外へは何も出ません", "关闭 · 没有任何内容离开本机"],
 'boot.jobs':["Jobs", "Trabajos", "ジョブ", "作业"],
 'boot.selected':["Selected", "Seleccionado", "選択中", "已选择"],
 'boot.map_label':["Model map: {{loaded}} loaded of {{count}} local, {{cloud}} cloud", "Mapa de modelos: {{loaded}} cargados de {{count}} locales, {{cloud}} en la nube", "モデルマップ: ローカル {{count}} 件中 {{loaded}} 件読み込み済み、クラウド {{cloud}} 件", "模型图：本地 {{count}} 个中已加载 {{loaded}} 个，云端 {{cloud}} 个"],
 'boot.legend':["Cube loaded · filled selected · orbit cloud · hollow idle", "Cubo cargado · relleno seleccionado · órbita nube · vacío inactivo", "キューブ 読み込み済み · 塗り 選択中 · 軌道 クラウド · 中空 待機", "立方体 已加载 · 实心 已选择 · 轨道 云端 · 空心 空闲"],
 'retrieval.bay_line':["Retrieve passages, rerank them and answer with citations, using three registered models.", "Recupera pasajes, los reordena y responde con citas, usando tres modelos registrados.", "登録済みの 3 つのモデルで、文章を検索し、並べ替え、引用付きで回答します。", "使用三个已注册的模型检索段落、重新排序并附引用作答。"],
 'broker.bay_line':["Providers, keys and routes. Requests to them leave this computer.", "Proveedores, claves y rutas. Las solicitudes a ellos salen de este equipo.", "プロバイダー、キー、経路。これらへのリクエストはこのコンピューターの外へ出ます。", "提供商、密钥和路由。发往它们的请求会离开本机。"],
 'broker.lede':["These routes leave this computer. Titanos sends a request to a provider only when you run one of its models, and only through an upstream enabled here.", "Estas rutas salen de este equipo. Titanos envía una solicitud a un proveedor solo cuando ejecuta uno de sus modelos y solo a través de un upstream habilitado aquí.", "これらの経路はこのコンピューターの外へ出ます。Titanos がプロバイダーへリクエストを送るのは、そのモデルを実行したときに、ここで有効にしたアップストリーム経由でのみです。", "这些路由会离开本机。只有在您运行某个提供商的模型时，Titanos 才会通过此处启用的上游向其发送请求。"],
 'broker.k_role':["Role", "Rol", "ロール", "角色"],
 'broker.k_policy':["Cloud policy", "Política de nube", "クラウドポリシー", "云端策略"],
 'broker.policy_value':["{{effective}} (file {{file}}, ceiling {{ceiling}})", "{{effective}} (archivo {{file}}, límite {{ceiling}})", "{{effective}}（ファイル {{file}}、上限 {{ceiling}}）", "{{effective}}（文件 {{file}}，上限 {{ceiling}}）"],
 'broker.k_vault':["Vault", "Almacén", "保管庫", "保管库"],
 'broker.k_upstreams':["Upstreams", "Upstreams", "アップストリーム", "上游"],
 'broker.upstream_counts':["{{enabled}} of {{total}} enabled · {{healthy}} healthy", "{{enabled}} de {{total}} habilitados · {{healthy}} en buen estado", "{{total}} 件中 {{enabled}} 件有効 · 正常 {{healthy}} 件", "{{total}} 个中已启用 {{enabled}} 个 · 健康 {{healthy}} 个"],
 'broker.k_egress':["Egress", "Salida", "送信先制御", "出站"],
 'broker.k_keyless':["Keyless use", "Uso sin clave", "キーなしの利用", "无密钥使用"],
 'broker.keyless_on':["Allowed on this computer", "Permitido en este equipo", "このコンピューターで許可", "本机允许"],
 'broker.keyless_off':["Needs a key with cloud access", "Requiere una clave con acceso a la nube", "クラウドアクセス権のあるキーが必要", "需要具有云端访问权限的密钥"],
 'broker.online':["Online", "En línea", "オンライン", "在线"],
 'broker.not_ready':["Not ready", "No disponible", "準備未完了", "未就绪"],
 'broker.breaker_open':["Breaker open", "Cortacircuito abierto", "ブレーカー作動中", "熔断已打开"],
 'broker.tile_meta':["upstream {{id}} · profile {{profile}}", "upstream {{id}} · perfil {{profile}}", "アップストリーム {{id}} · プロファイル {{profile}}", "上游 {{id}} · 配置 {{profile}}"],
 'broker.k_fingerprint':["Fingerprint", "Huella", "フィンガープリント", "指纹"],
 'broker.k_catalog':["Catalog", "Catálogo", "カタログ", "目录"],
 'broker.validated_value':["{{state}} · {{at}}", "{{state}} · {{at}}", "{{state}} · {{at}}", "{{state}} · {{at}}"],
 'broker.state_value':["{{enabled}} · breaker {{breaker}}", "{{enabled}} · cortacircuito {{breaker}}", "{{enabled}} · ブレーカー {{breaker}}", "{{enabled}} · 熔断 {{breaker}}"],
 'broker.catalog_value':["{{source}} · {{count}} models", "{{source}} · {{count}} modelos", "{{source}} · モデル {{count}} 件", "{{source}} · {{count}} 个模型"],
 'broker.vault_value_open':["{{backend}} · open", "{{backend}} · abierto", "{{backend}} · 開いています", "{{backend}} · 已打开"],
 'broker.vault_value_sealed':["{{backend}} · sealed", "{{backend}} · sellado", "{{backend}} · 封印中", "{{backend}} · 已封存"],
 'broker.k_credential':["Credential", "Credencial", "認証情報", "凭据"],
 'broker.credential_env':["{{id}} · env {{name}}", "{{id}} · env {{name}}", "{{id}} · 環境変数 {{name}}", "{{id}} · 环境变量 {{name}}"],
 'broker.credential_kind':["{{id}} · {{source}}", "{{id}} · {{source}}", "{{id}} · {{source}}", "{{id}} · {{source}}"],
 'broker.k_validation':["Validation", "Validación", "検証", "验证"],
 'broker.k_state':["State", "Estado", "状態", "状态"],
 'broker.k_models':["Models", "Modelos", "モデル", "模型"],
 'broker.model_count':["{{count}} listed", "{{count}} listados", "{{count}} 件", "已列出 {{count}} 个"],
 'broker.routes':["Cloud routes", "Rutas en la nube", "クラウド経路", "云端路由"],
 'broker.connect_title':["Connect a provider", "Conectar un proveedor", "プロバイダーを接続", "连接提供商"],
 'broker.connect_hint':["Pick a provider and give Titanos its key. A key found in this server's environment is used by name; its value never reaches this page.", "Elija un proveedor y entregue su clave a Titanos. Una clave encontrada en el entorno de este servidor se usa por su nombre; su valor nunca llega a esta página.", "プロバイダーを選び、そのキーを Titanos に渡します。このサーバーの環境にあるキーは名前で使われ、その値がこのページに届くことはありません。", "选择提供商并把它的密钥交给 Titanos。在此服务器环境中找到的密钥按名称使用，其值不会传到本页面。"],
 'broker.connect_cli_keyless':["This server does not accept provider setup from Studio without a key. On this computer run: titanos broker connect <provider> --from-env NAME (or --from-stdin to paste a key).", "Este servidor no acepta configurar proveedores desde Studio sin clave. En este equipo ejecute: titanos broker connect <provider> --from-env NAME (o --from-stdin para pegar una clave).", "このサーバーは、キーなしの Studio からのプロバイダー設定を受け付けません。このコンピューターで次を実行してください: titanos broker connect <provider> --from-env NAME（キーを貼り付ける場合は --from-stdin）。", "此服务器不接受无密钥的工作室设置提供商。请在本机运行：titanos broker connect <provider> --from-env NAME（或用 --from-stdin 粘贴密钥）。"],
 'broker.connect_cli_scope':["Connecting a provider needs a key with credentials:write. On this computer run: titanos broker connect <provider> --from-env NAME (or --from-stdin to paste a key).", "Conectar un proveedor requiere una clave con credentials:write. En este equipo ejecute: titanos broker connect <provider> --from-env NAME (o --from-stdin para pegar una clave).", "プロバイダーの接続には credentials:write を持つキーが必要です。このコンピューターで次を実行してください: titanos broker connect <provider> --from-env NAME（キーを貼り付ける場合は --from-stdin）。", "连接提供商需要具有 credentials:write 的密钥。请在本机运行：titanos broker connect <provider> --from-env NAME（或用 --from-stdin 粘贴密钥）。"],
 'broker.connect_provider':["Provider", "Proveedor", "プロバイダー", "提供商"],
 'broker.connect_option_done':["{{name}} · connected", "{{name}} · conectado", "{{name}} · 接続済み", "{{name}} · 已连接"],
 'broker.connect_option_env':["{{name}} · {{variable}} found", "{{name}} · {{variable}} encontrada", "{{name}} · {{variable}} あり", "{{name}} · 已找到 {{variable}}"],
 'broker.connect_update':["Connected as upstream {{id}}. Connecting again replaces its credential and refreshes its models.", "Conectado como upstream {{id}}. Volver a conectar reemplaza su credencial y actualiza sus modelos.", "アップストリーム {{id}} として接続済みです。再接続すると認証情報が置き換わり、モデル一覧が更新されます。", "已作为上游 {{id}} 连接。再次连接会替换其凭据并刷新其模型。"],
 'broker.connect_new':["Titanos creates the upstream from the provider profile, checks the key, lists its models and enables it.", "Titanos crea el upstream a partir del perfil del proveedor, comprueba la clave, lista sus modelos y lo habilita.", "Titanos はプロバイダーのプロファイルからアップストリームを作成し、キーを確認し、モデルを一覧にして有効にします。", "Titanos 会根据提供商配置创建上游、检查密钥、列出其模型并启用它。"],
 'broker.connect_source':["Key", "Clave", "キー", "密钥"],
 'broker.source_env_choice':["Use a key from this server's environment", "Usar una clave del entorno de este servidor", "このサーバーの環境にあるキーを使う", "使用此服务器环境中的密钥"],
 'broker.source_value_choice':["Paste a key (sent once, never shown again)", "Pegar una clave (se envía una vez y no se vuelve a mostrar)", "キーを貼り付ける（一度だけ送信され、再表示されません）", "粘贴密钥（只发送一次，不会再次显示）"],
 'broker.source_none_choice':["No key", "Sin clave", "キーなし", "无密钥"],
 'broker.source_none_hint':["This profile can run without a credential.", "Este perfil puede funcionar sin credencial.", "このプロファイルは認証情報なしで動作できます。", "此配置无需凭据即可运行。"],
 'broker.env_set':["{{name}} · set", "{{name}} · definida", "{{name}} · 設定済み", "{{name}} · 已设置"],
 'broker.env_unset':["{{name}} · not set", "{{name}} · sin definir", "{{name}} · 未設定", "{{name}} · 未设置"],
 'broker.env_placeholder':["Variable name, e.g. OPENAI_API_KEY", "Nombre de variable, p. ej. OPENAI_API_KEY", "変数名（例: OPENAI_API_KEY）", "变量名，例如 OPENAI_API_KEY"],
 'broker.env_hint_found':["Titanos stores a reference to {{name}}, never its value.", "Titanos guarda una referencia a {{name}}, nunca su valor.", "Titanos が保存するのは {{name}} への参照だけで、値は保存しません。", "Titanos 只保存对 {{name}} 的引用，从不保存其值。"],
 'broker.env_hint_missing':["None of these is set where Titanos runs. Set one and restart, or paste a key.", "Ninguna está definida donde se ejecuta Titanos. Defina una y reinicie, o pegue una clave.", "Titanos の実行環境ではいずれも設定されていません。設定して再起動するか、キーを貼り付けてください。", "Titanos 运行环境中均未设置。请设置一个并重启，或粘贴密钥。"],
 'broker.env_hint_none':["This provider has no well-known variable; type the name of one that is set where Titanos runs.", "Este proveedor no tiene una variable conocida; escriba el nombre de una definida donde se ejecuta Titanos.", "このプロバイダーには既定の変数がありません。Titanos の実行環境で設定済みの変数名を入力してください。", "此提供商没有常用变量；请输入 Titanos 运行环境中已设置的变量名。"],
 'broker.env_missing_route':["This server does not report environment keys (GET /api/broker/env-candidates answered {{status}}). Type a variable name, or paste a key.", "Este servidor no informa de claves del entorno (GET /api/broker/env-candidates respondió {{status}}). Escriba un nombre de variable o pegue una clave.", "このサーバーは環境のキーを報告しません（GET /api/broker/env-candidates の応答は {{status}}）。変数名を入力するか、キーを貼り付けてください。", "此服务器不报告环境密钥（GET /api/broker/env-candidates 返回 {{status}}）。请输入变量名或粘贴密钥。"],
 'broker.env_unavailable':["Environment keys could not be listed. Type a variable name, or paste a key.", "No se pudieron listar las claves del entorno. Escriba un nombre de variable o pegue una clave.", "環境のキーを一覧にできませんでした。変数名を入力するか、キーを貼り付けてください。", "无法列出环境密钥。请输入变量名或粘贴密钥。"],
 'broker.base_url':["Base URL", "URL base", "ベース URL", "基础 URL"],
 'broker.base_optional':["Base URL (optional)", "URL base (opcional)", "ベース URL（任意）", "基础 URL（可选）"],
 'broker.base_default':["Default: {{url}}", "Predeterminada: {{url}}", "既定: {{url}}", "默认：{{url}}"],
 'broker.base_required':["Required for this profile, e.g. https://host/v1", "Obligatoria para este perfil, p. ej. https://host/v1", "このプロファイルでは必須（例: https://host/v1）", "此配置必填，例如 https://host/v1"],
 'broker.connect_anonymous':["Let Studio and other keyless programs on this computer use this provider (otherwise a key with cloud access is needed)", "Permitir que Studio y otros programas sin clave de este equipo usen este proveedor (si no, se necesita una clave con acceso a la nube)", "このコンピューター上の Studio やキーなしのプログラムにこのプロバイダーの利用を許可する（許可しない場合はクラウドアクセス権のあるキーが必要）", "允许本机上的工作室和其他无密钥程序使用此提供商（否则需要具有云端访问权限的密钥）"],
 'broker.connect_aliases':["Also publish the provider's model ids as plain aliases", "Publicar también los id de modelo del proveedor como alias simples", "プロバイダーのモデル ID をそのまま別名としても公開する", "同时将提供商的模型 ID 发布为普通别名"],
 'broker.connect':["Connect", "Conectar", "接続", "连接"],
 'broker.connect_pick':["Pick a provider first.", "Elija primero un proveedor.", "先にプロバイダーを選んでください。", "请先选择提供商。"],
 'broker.connecting':["Connecting {{provider}}: checking the key and listing models…", "Conectando {{provider}}: comprobando la clave y listando modelos…", "{{provider}} を接続中: キーを確認し、モデルを一覧にしています…", "正在连接 {{provider}}：检查密钥并列出模型…"],
 'broker.connected':["Connected {{provider}}: {{count}} models found. They are listed under Models, marked External.", "{{provider}} conectado: {{count}} modelos encontrados. Aparecen en Modelos, marcados como Externo.", "{{provider}} を接続しました: モデル {{count}} 件。モデル一覧に「外部」として表示されます。", "已连接 {{provider}}：找到 {{count}} 个模型。它们列在模型中，标记为外部。"],
 'broker.connect_missing':["This server has no one-step connect (POST /api/broker/connect answered {{status}}). Update Titanos, or on this computer run: titanos broker connect {{provider}} --from-env NAME", "Este servidor no tiene conexión en un paso (POST /api/broker/connect respondió {{status}}). Actualice Titanos o, en este equipo, ejecute: titanos broker connect {{provider}} --from-env NAME", "このサーバーにはワンステップ接続がありません（POST /api/broker/connect の応答は {{status}}）。Titanos を更新するか、このコンピューターで次を実行してください: titanos broker connect {{provider}} --from-env NAME", "此服务器没有一步连接（POST /api/broker/connect 返回 {{status}}）。请更新 Titanos，或在本机运行：titanos broker connect {{provider}} --from-env NAME"],
 'broker.connect_unexpected':["The server answered, but not with a connect result; nothing is shown as connected.", "El servidor respondió, pero no con un resultado de conexión; no se muestra nada como conectado.", "サーバーは応答しましたが、接続結果ではありませんでした。接続済みとしては何も表示しません。", "服务器已响应，但不是连接结果；不会显示任何已连接的内容。"],
 'broker.connect_failed':["{{provider}} is not connected", "{{provider}} no está conectado", "{{provider}} は接続されていません", "{{provider}} 未连接"],
 'broker.connect_done':["{{provider}} connected", "{{provider}} conectado", "{{provider}} を接続しました", "{{provider}} 已连接"],
 'broker.k_upstream':["Upstream", "Upstream", "アップストリーム", "上游"],
 'broker.upstream_enabled':["{{id}} · {{state}}", "{{id}} · {{state}}", "{{id}} · {{state}}", "{{id}} · {{state}}"],
 'broker.credential_ref':["env {{name}} · {{fingerprint}}", "env {{name}} · {{fingerprint}}", "環境変数 {{name}} · {{fingerprint}}", "环境变量 {{name}} · {{fingerprint}}"],
 'broker.credential_vault':["{{source}} · {{fingerprint}}", "{{source}} · {{fingerprint}}", "{{source}} · {{fingerprint}}", "{{source}} · {{fingerprint}}"],
 'broker.policy_effective':["{{effective}} (ceiling {{ceiling}})", "{{effective}} (límite {{ceiling}})", "{{effective}}（上限 {{ceiling}}）", "{{effective}}（上限 {{ceiling}}）"],
 'broker.models_found':["{{count}} found", "{{count}} encontrados", "{{count}} 件", "找到 {{count}} 个"],
 'broker.models_more':["+{{count}} more", "+{{count}} más", "ほか {{count}} 件", "另有 {{count}} 个"],
 'broker.studio_policy':["Plain http to LAN upstreams is changed only from the CLI or with a key; edit broker.json on the host.", "El http sin cifrar hacia upstreams de la LAN solo se cambia desde la CLI o con una clave; edite broker.json en el equipo anfitrión.", "LAN 上のアップストリームへの平文 http は CLI かキーでのみ変更できます。ホスト上の broker.json を編集してください。", "到局域网上游的明文 http 只能通过命令行或密钥更改；请在主机上编辑 broker.json。"]
});
for(const [name,row] of Object.entries(workflowText)){en[name]=row[0];es[name]=row[1];ja[name]=row[2];zh[name]=row[3];}
function defaultStorage(){try{return globalThis.localStorage;}catch{return undefined;}}
const catalogs={en,es,ja,'zh-CN':zh},key='titanos.preferences.v1';let current={mode:'simple',locale:'en'},store=defaultStorage(),listening=false;
function matched(value){const text=String(value||'').toLowerCase();return text.startsWith('zh')?'zh-CN':text.startsWith('ja')?'ja':text.startsWith('es')?'es':'en';}
function read(storage,browserLanguage){let value;try{value=JSON.parse(storage?.getItem(key)||'null');}catch{}return{mode:value?.mode==='advanced'?'advanced':'simple',locale:locales.includes(value?.locale)?value.locale:matched(browserLanguage)};}
function write(){try{store?.setItem(key,JSON.stringify(current));}catch{}}
function documentState(){const root=globalThis.document?.documentElement;if(root){root.lang=current.locale;root.dataset.locale=current.locale;root.dataset.mode=current.mode;}globalThis.document?.dispatchEvent?.(new CustomEvent('titanos:preferences',{detail:{...current}}));}
function initialize(options={}){store=options.storage??defaultStorage();current=read(store,options.browserLanguage??globalThis.navigator?.language);if(!listening&&globalThis.addEventListener){listening=true;globalThis.addEventListener('storage',event=>{if(event.key!==key)return;current=read(store,globalThis.navigator?.language);documentState();});}documentState();return{...current};}
function setMode(mode){if(!['simple','advanced'].includes(mode))throw Error('Unsupported mode');current.mode=mode;write();documentState();return{...current};}
function setLocale(locale){if(!locales.includes(locale))throw Error('Unsupported locale');current.locale=locale;write();documentState();return{...current};}
function t(name,variables={}){const value=catalogs[current.locale]?.[name]??catalogs.en[name]??name;return value.replace(/\{\{(\w+)\}\}/g,(_,field)=>String(variables[field]??''));}
function formatBytes(value,locale=current.locale){if(!Number.isFinite(value)||value<0)return'';const units=['B','KB','MB','GB','TB'];let amount=value,index=0;while(amount>=1024&&index<units.length-1){amount/=1024;index++;}return`${new Intl.NumberFormat(locale==='en'?'en-US':locale==='es'?'es-ES':locale==='ja'?'ja-JP':'zh-CN',{maximumFractionDigits:amount<10&&index?1:0}).format(amount)} ${units[index]}`;}
return{locales,labels,catalogs,initialize,setMode,setLocale,state:()=>({...current}),t,formatBytes};
})();
