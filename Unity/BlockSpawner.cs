using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// Reads blocks.json produced by the MagicCamera Python pipeline
/// and spawns prefabs (or default cubes) at the specified positions.
/// </summary>
public class BlockSpawner : MonoBehaviour
{
    // ── JSON data classes ───────────────────────────────────────────────────

    [Serializable]
    public class BlockData
    {
        public int id;
        public string type;
        public float[] position;   // [x, y, z]  normalised
        public float[] rotation;   // [rx, ry, rz] Euler degrees
        public float[] scale;      // [sx, sy, sz] relative
        public float depth;
        public int area;
    }

    [Serializable]
    public class BlockList
    {
        public BlockData[] blocks;
    }

    // ── Prefab mapping (Inspector-friendly) ─────────────────────────────────

    [Serializable]
    public class PrefabEntry
    {
        public string typeName;
        public GameObject prefab;
    }

    // ── Inspector fields ────────────────────────────────────────────────────

    [Header("JSON Source")]
    [Tooltip("File name inside StreamingAssets (e.g. blocks.json)")]
    public string jsonFileName = "blocks.json";

    [Header("Prefab Mapping")]
    [Tooltip("Map object type names to prefabs. Unmatched types use the default.")]
    public List<PrefabEntry> prefabMap = new List<PrefabEntry>();

    [Tooltip("Fallback prefab for unmapped types. Leave empty to auto-create a cube.")]
    public GameObject defaultPrefab;

    [Header("World Settings")]
    [Tooltip("Multiplier applied to normalised positions so objects spread out in world space.")]
    public float worldScale = 5f;

    [Tooltip("Multiplier applied to object scale values.")]
    public float scaleMultiplier = 5f;

    [Tooltip("Automatically spawn all blocks on Start.")]
    public bool spawnOnStart = true;

    // ── Runtime state ───────────────────────────────────────────────────────

    private readonly List<GameObject> _spawnedObjects = new List<GameObject>();
    private Dictionary<string, GameObject> _prefabLookup;

    // ── Unity callbacks ─────────────────────────────────────────────────────

    void Start()
    {
        BuildPrefabLookup();

        if (spawnOnStart)
            SpawnAll();
    }

    // ── Public API ──────────────────────────────────────────────────────────

    /// <summary>Load blocks.json and spawn every block.</summary>
    public void SpawnAll()
    {
        ClearSpawned();

        string path = Path.Combine(Application.streamingAssetsPath, jsonFileName);

        if (!File.Exists(path))
        {
            Debug.LogError($"[BlockSpawner] JSON not found: {path}");
            return;
        }

        string json = File.ReadAllText(path);
        BlockList data = JsonUtility.FromJson<BlockList>(json);

        if (data == null || data.blocks == null || data.blocks.Length == 0)
        {
            Debug.LogWarning("[BlockSpawner] No blocks found in JSON.");
            return;
        }

        Debug.Log($"[BlockSpawner] Spawning {data.blocks.Length} blocks...");

        foreach (BlockData block in data.blocks)
            SpawnBlock(block);
    }

    /// <summary>Destroy all previously spawned objects.</summary>
    public void ClearSpawned()
    {
        foreach (GameObject obj in _spawnedObjects)
        {
            if (obj != null)
                Destroy(obj);
        }
        _spawnedObjects.Clear();
    }

    // ── Internal ────────────────────────────────────────────────────────────

    private void BuildPrefabLookup()
    {
        _prefabLookup = new Dictionary<string, GameObject>(StringComparer.OrdinalIgnoreCase);
        foreach (PrefabEntry entry in prefabMap)
        {
            if (!string.IsNullOrEmpty(entry.typeName) && entry.prefab != null)
                _prefabLookup[entry.typeName] = entry.prefab;
        }
    }

    private void SpawnBlock(BlockData block)
    {
        // Resolve prefab
        GameObject prefab = ResolvePrefab(block.type);
        bool isDefault = (prefab == null);

        GameObject obj;
        if (isDefault)
        {
            obj = GameObject.CreatePrimitive(PrimitiveType.Cube);
        }
        else
        {
            obj = Instantiate(prefab);
        }

        obj.name = $"Block_{block.id}_{block.type}";

        // Position
        Vector3 pos = ArrayToVector3(block.position) * worldScale;
        obj.transform.position = pos;

        // Rotation
        Vector3 rot = ArrayToVector3(block.rotation);
        obj.transform.rotation = Quaternion.Euler(rot);

        // Scale
        Vector3 scl = ArrayToVector3(block.scale) * scaleMultiplier;
        obj.transform.localScale = scl;

        // Parent under this object for tidiness
        obj.transform.SetParent(transform, true);

        // Colour default cubes by depth for quick visual feedback
        if (isDefault)
        {
            Renderer rend = obj.GetComponent<Renderer>();
            if (rend != null)
            {
                float t = Mathf.Clamp01(block.depth);
                rend.material.color = Color.Lerp(Color.red, Color.blue, t);
            }
        }

        _spawnedObjects.Add(obj);

        Debug.Log($"[BlockSpawner] [{block.id}] {block.type} → pos={pos} depth={block.depth:F3}");
    }

    private GameObject ResolvePrefab(string typeName)
    {
        if (_prefabLookup != null && _prefabLookup.TryGetValue(typeName, out GameObject prefab))
            return prefab;
        return defaultPrefab;
    }

    private static Vector3 ArrayToVector3(float[] arr)
    {
        if (arr == null || arr.Length < 3)
            return Vector3.zero;
        return new Vector3(arr[0], arr[1], arr[2]);
    }
}
