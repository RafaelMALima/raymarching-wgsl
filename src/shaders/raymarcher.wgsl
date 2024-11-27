const THREAD_COUNT = 16;
const PI = 3.1415927f;
const MAX_DIST = 1000.0;

@group(0) @binding(0)  
  var<storage, read_write> fb : array<vec4f>;

@group(1) @binding(0)
  var<storage, read_write> uniforms : array<f32>;

@group(2) @binding(0)
  var<storage, read_write> shapesb : array<shape>;

@group(2) @binding(1)
  var<storage, read_write> shapesinfob : array<vec4f>;

struct shape {
  transform : vec4f, // xyz = position
  radius : vec4f, // xyz = scale, w = global scale
  rotation : vec4f, // xyz = rotation
  op : vec4f, // x = operation, y = k value, z = repeat mode, w = repeat offset
  color : vec4f, // xyz = color
  animate_transform : vec4f, // xyz = animate position value (sin amplitude), w = animate speed
  animate_rotation : vec4f, // xyz = animate rotation value (sin amplitude), w = animate speed
  quat : vec4f, // xyzw = quaternion
  transform_animated : vec4f, // xyz = position buffer
};

struct march_output {
  color : vec3f,
  depth : f32,
  outline : bool,
};

fn op_smooth_union(d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f
{
  var h = clamp(0.5 + 0.5 * (d2 - d1) / max(k, 0.0001), 0.0, 1.0);
  return vec4f(mix(col2, col1, h), mix(d2, d1, h) - k * h * (1.0 - h));}

fn op_smooth_subtraction(d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f
{
  var h = clamp(0.5 - 0.5 * (d2 + d1) / max(k, 0.0001), 0.0, 1.0);
  return vec4f(mix(col2, col1, h), mix(d2, -d1, h) + k * h * (1.0 - h));
}

fn op_smooth_intersection(d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f
{
  var h = clamp(0.5 - 0.5 * (d2 - d1) / max(k, 0.0001), 0.0, 1.0);
  return vec4f(mix(col2, col1, h), mix(d2, d1, h) + k * h * (1.0 - h));
}

fn op(op: f32, d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f
{
  // union
  if (op < 1.0)
  {
    return op_smooth_union(d1, d2, col1, col2, k);
  }

  // subtraction
  if (op < 2.0)
  {
    return op_smooth_subtraction(d1, d2, col1, col2, k);
  }

  // intersection
  return op_smooth_intersection(d2, d1, col2, col1, k);
}

fn repeat(p: vec3f, offset: vec3f) -> vec3f
{
  return modc(p + offset * 0.5, offset) - offset * 0.5;
}

fn transform_p(p: vec3f, option: vec2f) -> vec3f
{
  // normal mode
  if (option.x <= 1.0)
  {
    return p;
  }

  // return repeat / mod mode
  return repeat(p, vec3f(option.y));
}

fn scene(p: vec3f) -> vec4f // xyz = color, w = distance
{
  //braga, c vc ta lendo isso, existe algum motivo pra voce declarar todas as variaveis como var? mudando porque me incomoda profundamente, mesmo mudando nada
  let mandelbulb = uniforms[18];
  let weird_thing = uniforms[19];
  var d = mix(100.0, p.y, uniforms[17]);

  let spheresCount = i32(uniforms[2]);
  let boxesCount = i32(uniforms[3]);
  let torusCount = i32(uniforms[4]);

  let all_objects_count = spheresCount + boxesCount + torusCount;
  var result = vec4f(vec3f(1.0), d);

  var c1 : vec3f;
  var c2 : vec3f;
  var count = 0;
  for (var i = 0; i < all_objects_count; i = i + 1)
  {
    // get shape and shape order (shapesinfo)
    let index_ = i32(shapesinfob[i].y);
    let shape_type = shapesinfob[i].x; 
    var info = shapesb[index_];

    let p_transform = transform_p( p - info.transform_animated.xyz,info.op.zw);
    if ( shape_type > 1.0 ) // torus
    { 
      d = sdf_torus(p_transform,info.radius.xy,info.quat); 
    }
    else if (shape_type > 0.0)// caixa
    {
      d = sdf_round_box(p_transform, info.radius.xyz, info.radius.w, info.quat);
    } 
    else  // bola
    {
      d = sdf_sphere(p_transform,info.radius,info.quat);
    }

    //desgraca de union sub e sla o q. sim ta ilegivel fazer o que
    result = op(info.op.x,d,result.w,info.color.xyz,result.xyz,info.op.y);
  }
  return result;
}

fn march(ro: vec3f, rd: vec3f) -> march_output
{
  let outline = uniforms[26];
  let outline_width = uniforms[27];
  let max_marching_steps = i32(uniforms[5]);
  let EPSILON = uniforms[23];
  let march_step = uniforms[22];

  var depth = 0.0;
  var distance = 10000.;
  var min_dist = MAX_DIST;
  for (var depth = 0.0; depth < f32(MAX_DIST); depth+=distance)
  {
    // call scene function and march
    let ray_info = scene(ro+rd*depth);
    distance = ray_info.w;

    if (ray_info.w < min_dist)
    {
      min_dist = ray_info.w;
      if (min_dist < EPSILON ){
        return march_output(ray_info.xyz,depth,false);
      }
    }      
    // if the depth is greater than the max distance or the distance is less than the epsilon, break
    else if(depth > MAX_DIST)
    {
      break;
    }
  }
  if (outline > 0 && min_dist < outline_width){
    return march_output(vec3f(0.0), MAX_DIST, true);
  }
  return march_output(vec3f(0.0), MAX_DIST, false);
}

fn get_normal(p: vec3f) -> vec3f
{
  let epsilon = 0.0001;
  let nx = scene(p + vec3f(epsilon,0,0)).w - scene(p - vec3f(epsilon,0,0)).w;
  let ny = scene(p + vec3f(0,epsilon,0)).w - scene(p - vec3f(0,epsilon,0)).w;
  let nz = scene(p + vec3f(0,0,epsilon)).w - scene(p - vec3f(0,0,epsilon)).w;
  return normalize(vec3f(nx, ny, nz));
}

//portando do glsl pq nn existe essa bagaca no wgsl
fn map(ro:vec3f, rd:vec3f, t:f32) -> f32{
  return scene(ro + rd*t).w;
}

// https://iquilezles.org/articles/rmshadows/
fn get_soft_shadow(ro: vec3f, rd: vec3f, tmin: f32, tmax: f32, k: f32) -> f32
{
  var res = 1.0;
  var t = tmin;
  for( var i=0.; i<256. && t<tmax; i+=1)
  {
    let h = map(ro, rd, t);
    res = min(res, k * h / t);
    t += clamp(h, 0.005, 0.50);
    if( res<-1.0 || t>tmax ) { break; }
  }
  res = max(res,-1.0);
  return 0.25*(1.0+res)*(1.0+res)*(2.0-res);
}





fn get_AO(current: vec3f, normal: vec3f) -> f32
{
  var occ = 0.0;
  var sca = 1.0;
  for (var i = 0; i < 5; i = i + 1)
  {
    var h = 0.001 + 0.15 * f32(i) / 4.0;
    var d = scene(current + h * normal).w;
    occ += (h - d) * sca;
    sca *= 0.95;
  }

  return clamp( 1.0 - 2.0 * occ, 0.0, 1.0 ) * (0.5 + 0.5 * normal.y);
}

fn get_ambient_light(light_pos: vec3f, sun_color: vec3f, rd: vec3f) -> vec3f
{
  var backgroundcolor1 = int_to_rgb(i32(uniforms[12]));
  var backgroundcolor2 = int_to_rgb(i32(uniforms[29]));
  var backgroundcolor3 = int_to_rgb(i32(uniforms[30]));
  
  var ambient = backgroundcolor1 - rd.y * rd.y * 0.5;
  ambient = mix(ambient, 0.85 * backgroundcolor2, pow(1.0 - max(rd.y, 0.0), 4.0));

  var sundot = clamp(dot(rd, normalize(vec3f(light_pos))), 0.0, 1.0);
  var sun = 0.25 * sun_color * pow(sundot, 5.0) + 0.25 * vec3f(1.0,0.8,0.6) * pow(sundot, 64.0) + 0.2 * vec3f(1.0,0.8,0.6) * pow(sundot, 512.0);
  ambient += sun;
  ambient = mix(ambient, 0.68 * backgroundcolor3, pow(1.0 - max(rd.y, 0.0), 16.0));

  return ambient;
}

fn get_light(current: vec3f, obj_color: vec3f, rd: vec3f) -> vec3f
{
  let light_position = vec3f(uniforms[13], uniforms[14], uniforms[15]);
  let sunlight_color = int_to_rgb(i32(uniforms[16]));
  let ambient_light = get_ambient_light(light_position, sunlight_color, rd);
  let surface_normal = get_normal(current);
  let light_direction = normalize(light_position - current);
  let diffuse_intensity = max(dot(surface_normal, light_direction), 0.);
  let epsilon = 0.0001;
  let shadow_factor = get_soft_shadow(current + surface_normal * epsilon, light_direction, epsilon, length(light_position - current), 32.);
  let diffuse_lighting = diffuse_intensity * obj_color  * sunlight_color * shadow_factor;
  let ambient_occlusion = get_AO(current, surface_normal);
  var final_color = ambient_light * obj_color ;
  final_color += diffuse_lighting;
  final_color *= ambient_occlusion;
  final_color = clamp(final_color, vec3f(0.0), vec3f(1.0));
  return final_color;
}


fn set_camera(ro: vec3f, ta: vec3f, cr: f32) -> mat3x3<f32>
{
  var cw = normalize(ta - ro);
  var cp = vec3f(sin(cr), cos(cr), 0.0);
  var cu = normalize(cross(cw, cp));
  var cv = normalize(cross(cu, cw));
  return mat3x3<f32>(cu, cv, cw);
}

fn animate(val: vec3f, amplitude: vec3f, speed: f32, time: f32) -> vec3f {
  return val + amplitude * sin(speed * time);
}

@compute @workgroup_size(THREAD_COUNT, 1, 1)
fn preprocess(@builtin(global_invocation_id) id : vec3u) {
  var time = uniforms[0];
  var spheresCount = i32(uniforms[2]);
  var boxesCount = i32(uniforms[3]);
  var torusCount = i32(uniforms[4]);
  var all_objects_count = spheresCount + boxesCount + torusCount;

  if (i32(id.x) >= all_objects_count) {
    return;
  }

  let idx = i32(id.x);
  let shape_info = shapesb[idx];

  // transform
  let transform_new = animate(shape_info.transform.xyz, shape_info.animate_transform.xyz, shape_info.animate_transform.w, time);
  shapesb[idx].transform_animated = vec4f(transform_new, shape_info.transform.w);

  // rotation
  let quaternion = animate(shape_info.rotation.xyz, shape_info.animate_rotation.xyz, shape_info.animate_rotation.w, time);
  shapesb[idx].quat = quaternion_from_euler(quaternion);
}

@compute @workgroup_size(THREAD_COUNT, THREAD_COUNT, 1)
fn render(@builtin(global_invocation_id) id: vec3u)
{
  // Unpack data
  let fragCoord = vec2f(f32(id.x), f32(id.y));
  let rez = vec2f(uniforms[1]);
  let time = uniforms[0];
  let outline_color = uniforms[28];

  // Camera setup
  let lookfrom = vec3f(uniforms[6], uniforms[7], uniforms[8]);
  let lookat = vec3f(uniforms[9], uniforms[10], uniforms[11]);
  let camera = set_camera(lookfrom, lookat, 0.0);
  let ro = lookfrom;

  // Get ray direction
  var uv = (fragCoord - 0.5 * rez) / rez.y;
  uv.y = -uv.y;
  let rd = camera * normalize(vec3f(uv, 1.0));

  // Call march function and get the color/depth
  let m_out = march(ro, rd);
  let depth = m_out.depth;

  var final_color : vec3f;
  if (depth < MAX_DIST){
    let obj_color = get_light(ro + rd * depth, m_out.color, rd);
    final_color = linear_to_gamma(obj_color);
    fb[mapfb(id.xy, uniforms[1])] = vec4f(final_color, 1.0);
    return;
  }
  
  else {
    let light_position = vec3f(uniforms[13], uniforms[14], uniforms[15]);
    let sun_color = int_to_rgb(i32(uniforms[16]));
    let ambient_color = get_ambient_light(light_position, sun_color, rd);
    final_color = linear_to_gamma(ambient_color);
  }

  if (m_out.outline){
    final_color = linear_to_gamma(vec3f(outline_color));
  }

  fb[mapfb(id.xy, uniforms[1])] = vec4f(final_color, 1.0);
}