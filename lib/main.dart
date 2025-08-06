import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Quick sanity check that assets are bundled
  final manifest = await rootBundle.loadString('AssetManifest.json');
  if (!manifest.contains('assets/recycle_classifier.tflite')) {
    debugPrint('⚠️ recycle_classifier.tflite not found in assets');
  }
  if (!manifest.contains('assets/labels.txt')) {
    debugPrint('⚠️ labels.txt not found in assets');
  }

  runApp(const WasteClassifierApp());
}

class WasteClassifierApp extends StatelessWidget {
  const WasteClassifierApp({super.key});

  @override
  Widget build(BuildContext context) {
    final seed = const Color(0xFF1B5E20); // deep green
    return MaterialApp(
      title: 'Waste Classifier',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: seed,
          brightness: Brightness.light,
        ),
        textTheme: GoogleFonts.poppinsTextTheme(),
        useMaterial3: true,
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            elevation: 2,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(14),
            ),
            padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 14),
          ),
        ),
      ),
      home: const ClassifierHomePage(),
    );
  }
}

class ClassifierHomePage extends StatefulWidget {
  const ClassifierHomePage({super.key});
  @override
  State<ClassifierHomePage> createState() => _ClassifierHomePageState();
}

class _ClassifierHomePageState extends State<ClassifierHomePage>
    with SingleTickerProviderStateMixin {
  Interpreter? _interpreter;
  List<String>? _labels;

  File? _image;
  bool _working = false;

  // Display states
  String _displayTitle = ''; // e.g., "Plastic (Recyclable) 85%"
  String _category = '';     // "Recyclable" | "Non Recyclable" | "Check Condition"
  double _confidence = 0.0;  // 0..1
  List<String> _recyclePoints = [];

  final ImagePicker _picker = ImagePicker();
  final List<String> _history = [];

  late final AnimationController _controller =
  AnimationController(vsync: this, duration: const Duration(milliseconds: 450));
  late final Animation<double> _fade =
  CurvedAnimation(parent: _controller, curve: Curves.easeIn);

  // Preprocessing – set to true if you trained with [-1,1] scaling before the backbone.
  static const bool useExternalMobilenetPreprocessing = true;

  // Will be overwritten by model input shape after load (e.g., 224x224x3)
  int _inputH = 224, _inputW = 224, _inputC = 3;

  /// Guidance bullets (keys should match labels.txt – we normalize lookups).
  static const Map<String, List<String>> recycleMap = {
    'alluminium': [
      'Recycled into aluminum ingots',
      'Used for beverage cans',
      'Auto parts and machinery components',
    ],
    'aluminium': [
      'Recycled into aluminum ingots',
      'Used for beverage cans',
      'Auto parts and machinery components',
    ],
    'cardboard': [
      'Recycled into new packaging board',
      'Shipping and storage boxes',
      'Paperboard for retail packaging',
    ],
    'glass': [
      'New bottles and jars',
      'Fiberglass insulation',
      'Decorative tiles and glassware',
    ],
    'paper': [
      'Newsprint and office paper',
      'Tissue and paper towels',
      'Packaging and cardboard backing',
    ],
    'plastic': [
      'Bottles and containers',
      'Textile fibers (clothing/carpets)',
      'Packaging films and wraps',
    ],
    'diaper': [
      'Not recyclable (contamination)',
      'Dispose in trash per local guidance',
    ],
    'organic waste': [
      'Compost for soil enrichment',
      'Biogas via anaerobic digestion',
    ],
    'organic_waste': [
      'Compost for soil enrichment',
      'Biogas via anaerobic digestion',
    ],
    'pizza_box': [
      'Greasy parts → trash',
      'Clean sections → paper recycling',
    ],
    'styrofoam': [
      'Not recyclable curbside in most areas',
      'Check special drop-off facilities',
    ],
    'tissue': [
      'Usually not recyclable (contamination/short fibers)',
      'Trash or compost if unsoiled and allowed',
    ],
  };

  /// Which labels are *generally* recyclable vs non-recyclable
  final Set<String> _recyclableSet = const {
    'alluminium', 'aluminium', 'cardboard', 'glass', 'paper', 'plastic',
  };
  final Set<String> _nonRecyclableSet = const {
    'diaper', 'organic waste', 'organic_waste', 'styrofoam', 'tissue',
    // pizza_box is conditional — handled separately
  };

  @override
  void initState() {
    super.initState();
    _initializeAll();
  }

  Future<void> _initializeAll() async {
    await _loadModelAndLabels();
    await _loadHistory();
  }

  Future<void> _loadModelAndLabels() async {
    try {
      // Verify asset exists (size)
      final byteData = await rootBundle.load('assets/recycle_classifier.tflite');
      debugPrint('✅ Model size: ${byteData.lengthInBytes} bytes');

      // Load interpreter (first try with explicit prefix)
      _interpreter = await Interpreter.fromAsset('assets/recycle_classifier.tflite');
    } catch (_) {
      debugPrint('ℹ️ Retrying model load without "assets/" prefix…');
      _interpreter = await Interpreter.fromAsset('recycle_classifier.tflite');
    }

    // Discover input/output shapes
    final inputShape = _interpreter!.getInputTensor(0).shape; // [1,H,W,3]
    if (inputShape.length == 4) {
      _inputH = inputShape[1];
      _inputW = inputShape[2];
      _inputC = inputShape[3];
    }
    debugPrint('🧠 TFLite input: $inputShape  -> H=$_inputH W=$_inputW C=$_inputC');
    debugPrint('🧠 TFLite output: ${_interpreter!.getOutputTensor(0).shape}');

    // Load labels
    final raw = await rootBundle.loadString('assets/labels.txt');
    _labels = raw.split('\n').map((e) => e.trim()).where((e) => e.isNotEmpty).toList();
    debugPrint('✅ Labels loaded: ${_labels!.length}');
  }

  Future<void> _loadHistory() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _history
        ..clear()
        ..addAll(prefs.getStringList('history') ?? []);
    });
  }

  Future<void> _saveToHistory(String entry) async {
    final prefs = await SharedPreferences.getInstance();
    _history.insert(0, entry);
    if (_history.length > 30) _history.removeLast();
    await prefs.setStringList('history', _history);
  }

  // ===== UI actions =====

  Future<void> _pickImage(ImageSource src) async {
    if (_working) return;
    setState(() {
      _working = true;
      _image = null;
      _displayTitle = '';
      _category = '';
      _confidence = 0;
      _recyclePoints = [];
    });

    try {
      if (src == ImageSource.camera) {
        final status = await Permission.camera.status;
        if (status.isDenied || status.isPermanentlyDenied) {
          final newStatus = await Permission.camera.request();
          if (!newStatus.isGranted) {
            setState(() => _displayTitle = 'Camera permission denied.');
            if (newStatus.isPermanentlyDenied) {
              await openAppSettings();
            }
            return;
          }
        }
      }

      final XFile? x = await _picker.pickImage(
        source: src,
        maxWidth: 1200,
        maxHeight: 1200,
        imageQuality: 92,
      );
      if (x == null) {
        setState(() {
          _displayTitle = 'No image selected.';
          _working = false;
        });
        return;
      }

      final file = File(x.path);
      setState(() => _image = file);
      await _classify(file);
    } catch (e, st) {
      debugPrint('⚠️ pick error: $e\n$st');
      setState(() => _displayTitle = 'Error picking image.');
    } finally {
      setState(() => _working = false);
    }
  }

  Future<void> _classify(File file) async {
    if (_interpreter == null || _labels == null) {
      setState(() => _displayTitle = 'Model or labels not loaded.');
      return;
    }

    try {
      final input = await _preprocess(file, _inputW, _inputH);
      final output = List.filled(_labels!.length, 0.0).reshape([1, _labels!.length]);
      _interpreter!.run(input, output);

      final scores = (output[0] as List<double>);
      var bestIdx = 0;
      var best = -1.0;
      for (var i = 0; i < scores.length; i++) {
        if (scores[i] > best) {
          best = scores[i];
          bestIdx = i;
        }
      }

      final raw = _labels![bestIdx];
      final normKey = raw.toLowerCase().trim();
      final pretty = raw
          .replaceAll('_', ' ')
          .split(' ')
          .where((w) => w.isNotEmpty)
          .map((w) => w[0].toUpperCase() + w.substring(1).toLowerCase())
          .join(' ');

      // Category
      String category;
      if (_recyclableSet.contains(normKey)) {
        category = 'Recyclable';
      } else if (_nonRecyclableSet.contains(normKey)) {
        category = 'Non Recyclable';
      } else if (normKey == 'pizza_box') {
        category = 'Check Condition';
      } else {
        category = 'Unknown';
      }

      // Guidance bullets
      final lookupKeys = {
        normKey,
        normKey.replaceAll('_', ' '),
        normKey.replaceAll(' ', '_'),
      };
      List<String> bullets = [];
      for (final k in lookupKeys) {
        if (recycleMap.containsKey(k)) {
          bullets = recycleMap[k]!;
          break;
        }
      }
      if (bullets.isEmpty) {
        bullets = best < 0.5
            ? ['Low confidence — please retake a clearer photo.']
            : ['No specific guidance found for this class.'];
      }

      final pct = (best * 100).clamp(0, 100).toStringAsFixed(0);
      final title = '$pretty (${category}) $pct%'; // <-- "Plastic (Recyclable) 85%"

      setState(() {
        _displayTitle = title;
        _category = category;
        _confidence = best.clamp(0.0, 1.0);
        _recyclePoints = bullets;
      });

      _controller.forward(from: 0);
      await _saveToHistory(title);
    } catch (e, st) {
      debugPrint('⚠️ inference error: $e\n$st');
      setState(() => _displayTitle = 'Error during inference.');
    }
  }

  /// Preprocess to [1, H, W, 3].
  Future<List<List<List<List<double>>>>> _preprocess(File f, int w, int h) async {
    final bytes = await f.readAsBytes();
    final codec = await ui.instantiateImageCodec(bytes, targetWidth: w, targetHeight: h);
    final frame = await codec.getNextFrame();
    final img = frame.image;
    final bd = await img.toByteData(format: ui.ImageByteFormat.rawRgba);
    final pixels = bd!.buffer.asUint8List();

    return [
      List.generate(h, (y) {
        return List.generate(w, (x) {
          final i = (y * w + x) * 4;
          final r = pixels[i].toDouble();
          final g = pixels[i + 1].toDouble();
          final b = pixels[i + 2].toDouble();

          if (useExternalMobilenetPreprocessing) {
            // [0,255] -> [-1,1]
            return [(r / 127.5) - 1.0, (g / 127.5) - 1.0, (b / 127.5) - 1.0];
          } else {
            // [0,255] -> [0,1]
            return [r / 255.0, g / 255.0, b / 255.0];
          }
        });
      })
    ];
  }

  // ===== UI helpers =====

  Color _categoryColor(BuildContext context) {
    switch (_category) {
      case 'Recyclable':
        return Colors.green.shade600;
      case 'Non Recyclable':
        return Colors.red.shade600;
      case 'Check Condition':
        return Colors.amber.shade700;
      default:
        return Theme.of(context).colorScheme.primary;
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    _interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final catColor = _categoryColor(context);

    return Scaffold(
      backgroundColor: theme.colorScheme.background,
      appBar: AppBar(
        elevation: 2,
        title: Text(
          'Waste Classifier',
          style: (theme.textTheme.titleLarge ?? const TextStyle()).copyWith(
            color: theme.colorScheme.onPrimary,
            fontWeight: FontWeight.w600,
          ),
        ),
        backgroundColor: theme.colorScheme.primary,
        actions: [
          _history.isNotEmpty
              ? IconButton(
            tooltip: 'History',
            icon: const Icon(Icons.history),
            onPressed: () => _showHistory(context),
          )
              : const SizedBox.shrink(),
          const SizedBox(width: 8),
        ],
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            // Image area
            ClipRRect(
              borderRadius: BorderRadius.circular(18),
              child: _image != null
                  ? Image.file(
                _image!,
                height: 260,
                width: double.infinity,
                fit: BoxFit.cover,
              )
                  : Container(
                height: 260,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(18),
                  gradient: LinearGradient(
                    colors: [
                      theme.colorScheme.surfaceVariant,
                      theme.colorScheme.surface,
                    ],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                ),
                child: Icon(Icons.image_outlined,
                    size: 64, color: Colors.grey.shade500),
              ),
            ),
            const SizedBox(height: 16),

            // Result card
            _displayTitle.isNotEmpty
                ? FadeTransition(
              opacity: _fade,
              child: Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(16),
                  color: theme.colorScheme.surface,
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.05),
                      blurRadius: 10,
                      offset: const Offset(0, 4),
                    )
                  ],
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Title
                    Text(
                      _displayTitle,
                      style: (theme.textTheme.titleMedium ?? const TextStyle())
                          .copyWith(fontWeight: FontWeight.w700),
                    ),
                    const SizedBox(height: 10),

                    // Category chip + confidence bar
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 10, vertical: 6),
                          decoration: BoxDecoration(
                            color: catColor.withOpacity(0.12),
                            borderRadius: BorderRadius.circular(999),
                            border: Border.all(color: catColor, width: 1),
                          ),
                          child: Text(
                            _category.isEmpty ? '—' : _category,
                            style: (theme.textTheme.labelLarge ?? const TextStyle())
                                .copyWith(
                              color: catColor,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: ClipRRect(
                            borderRadius: BorderRadius.circular(6),
                            child: LinearProgressIndicator(
                              value: _confidence.clamp(0.0, 1.0),
                              minHeight: 8,
                              backgroundColor:
                              theme.colorScheme.surfaceVariant,
                              color: catColor,
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),

                    if (_recyclePoints.isNotEmpty)
                      ..._recyclePoints.map((p) {
                        return Padding(
                          padding: const EdgeInsets.symmetric(vertical: 3),
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text('• ',
                                  style: (theme.textTheme.bodyLarge ??
                                      const TextStyle())
                                      .copyWith(
                                      color: theme
                                          .colorScheme.onSurface)),
                              Expanded(
                                child: Text(
                                  p,
                                  style: (theme.textTheme.bodyLarge ??
                                      const TextStyle())
                                      .copyWith(
                                    fontStyle: FontStyle.italic,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        );
                      }),
                  ],
                ),
              ),
            )
                : const SizedBox.shrink(),

            const SizedBox(height: 20),

            // Buttons
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.gallery),
                    icon: const Icon(Icons.photo_library_outlined),
                    label: const Text('Gallery'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: theme.colorScheme.secondary,
                      foregroundColor: theme.colorScheme.onSecondary,
                    ),
                  ),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt_outlined),
                    label: const Text('Camera'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: theme.colorScheme.primary,
                      foregroundColor: theme.colorScheme.onPrimary,
                    ),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 16),
            _working
                ? Center(
              child: CircularProgressIndicator(
                  color: theme.colorScheme.primary),
            )
                : const SizedBox.shrink(),
          ],
        ),
      ),
    );
  }

  void _showHistory(BuildContext context) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Theme.of(context).colorScheme.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(18)),
      ),
      builder: (_) => DraggableScrollableSheet(
        initialChildSize: 0.6,
        expand: false,
        builder: (_, ctl) => ListView.builder(
          controller: ctl,
          padding: const EdgeInsets.symmetric(vertical: 12),
          itemCount: _history.length,
          itemBuilder: (_, i) => ListTile(
            leading: Icon(Icons.history,
                color: Theme.of(context).colorScheme.primary),
            title: Text(_history[i]),
          ),
        ),
      ),
    );
  }
}
